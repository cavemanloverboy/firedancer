#ifndef HEADER_fd_src_ballet_ed25519_avx512_fd_ed25519_x8_h
#define HEADER_fd_src_ballet_ed25519_avx512_fd_ed25519_x8_h

/* Lane-per-signature Ed25519 batch verification.

   fd_ed25519_verify() spends nearly all of its time in one dependent chain
   of field operations for one signature.  fd_r43x6 makes that chain as
   short as it can by spreading a single field element over the lanes of a
   ZMM register, but there is a floor: a 256-bit scalar multiplication is
   ~250 sequential point doublings no matter how wide the multiplier is.

   This path takes the other axis.  When a caller has >=8 unrelated
   signatures on hand -- which is the normal case for a validator draining a
   sigverify queue -- it puts ONE SIGNATURE IN EACH AVX-512 LANE and runs
   eight verifications in lockstep.  Every VPMADD52 then retires eight
   signatures' worth of field arithmetic, and the sequential depth per
   signature is unchanged.  Concretely, relative to calling
   fd_ed25519_verify() in a loop:

     - field arithmetic is fd_r51x8 (see fd_r51x8.h), 8 independent
       elements per vector rather than 1 element per vector

     - the SHA-512 challenge hashes go through fd_sha512_batch, which is
       already an 8-lane AVX-512 multi-buffer implementation

     - the signature point R is never decompressed.  Instead the affine y
       of the recomputed point is compared against R's y and its x sign bit
       against R's sign bit, which is the same predicate (see below) and
       saves one inverse-square-root -- about 265 field multiplies -- per
       signature

     - the one modular inversion each signature needs to reach affine
       coordinates is replaced by a Montgomery batch inversion across the
       whole chunk, so the batch pays ~3 multiplies per signature plus a
       single inversion per 64

   ACCEPTANCE PREDICATE.  This is bit-for-bit the predicate implemented by
   fd_ed25519_verify() in fd_ed25519_user.c, including its deliberate
   deviations from ed25519-dalek 4.x:

     - S must be canonically reduced (S < L)
     - A is decoded permissively; a non-canonical A encoding is ACCEPTED
     - R is likewise treated permissively; a non-canonical R encoding is
       ACCEPTED, matching fd_ed25519_verify's point comparison rather than
       dalek 4.x's compressed-byte comparison
     - small order A or small order R is REJECTED (verify_strict)
     - the cofactorless equation [S]B == R + [k]A is checked

   The small-order test is done on the encodings rather than on decoded
   points.  That is exact, not an approximation: the permissive decoder
   reduces y modulo p and the sign bit does not change a point's order, so
   the encodings that decode to one of the eight small-order points are
   exactly the seven low-255-bit values {0, 1, p-1, p, p+1, +/-alpha},
   where alpha is the y of the order-8 points.

   Likewise, skipping R's decompression does not change the verdict.  For a
   decoded R, point equality against the recomputed point is equivalent to
   equality of affine y together with equality of the x sign bit, and an R
   that does not decode cannot match any curve point's (y, sign(x)) pair.

   VARIABLE TIME.  Table lookups are indexed by public scalar digits and
   the verdict masks branch on public data.  This is a verification-only
   path.  Do not use it for signing or for anything with a secret scalar. */

#include "../fd_ed25519.h"

#if FD_HAS_AVX512

FD_PROTOTYPES_BEGIN

/* fd_ed25519_verify_batch_x8 verifies cnt independent signatures and writes
   one independent verdict per signature into ok[].

   msgs[i]/msg_sz[i] is the i-th message, sigs[i] its 64-byte signature and
   pubs[i] its 32-byte public key.  ok[i] is set to FD_ED25519_SUCCESS or to
   an FD_ED25519_ERR_* code.

   ok[i]==FD_ED25519_SUCCESS if and only if fd_ed25519_verify() would have
   returned FD_ED25519_SUCCESS -- that equivalence is exact and is what the
   differential test asserts.  The failure CLASSIFICATION differs in exactly
   one case: when R is not a decodable point encoding, fd_ed25519_verify()
   notices while decompressing R and reports FD_ED25519_ERR_SIG, whereas
   this path never decompresses R and reports FD_ED25519_ERR_MSG.  Callers
   that only test against FD_ED25519_SUCCESS -- which is what the sigverify
   tile does -- see no difference at all.

   Returns FD_ED25519_SUCCESS if every signature verified and
   FD_ED25519_ERR_SIG otherwise; ok[] is authoritative either way.  cnt==0
   is fine.  There is no batch size limit and no requirement that cnt be a
   multiple of 8; the tail is handled by padding the unused lanes with
   inert work.

   Messages up to FD_ED25519_X8_FAST_MSG_MAX bytes are hashed through the
   8-lane multi-buffer SHA-512.  Longer messages fall back to the scalar
   streaming SHA-512 for the hash only; the curve arithmetic stays 8-lane.

   This function is thread safe and allocation free.  It uses a few KiB of
   stack. */

int
fd_ed25519_verify_batch_x8( uchar const * const * msgs,
                            ulong         const * msg_sz,
                            uchar const * const * sigs,
                            uchar const * const * pubs,
                            int *                 ok,
                            ulong                 cnt );

/* Messages at or below this size are staged into an on-stack buffer so the
   challenge hash can go through the 8-lane multi-buffer SHA-512.  1232 is
   the Solana transaction MTU, so the whole normal workload fits. */

#define FD_ED25519_X8_FAST_MSG_MAX (1408UL)

FD_PROTOTYPES_END

#endif /* FD_HAS_AVX512 */
#endif /* HEADER_fd_src_ballet_ed25519_avx512_fd_ed25519_x8_h */
