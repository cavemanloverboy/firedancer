/* Differential test and benchmark for the lane-per-signature batch
   verifier.  fd_ed25519_verify is the oracle: for every input the x8 path
   must return exactly the same verdict, including the error code. */

#include "../../fd_ballet.h"
#include "../fd_ed25519.h"
#include "fd_ed25519_x8.h"

#if !FD_HAS_AVX512
int main( int argc, char ** argv ) {
  fd_boot( &argc, &argv );
  FD_LOG_WARNING(( "skip: no AVX-512" ));
  fd_halt();
  return 0;
}
#else

#include "../fd_curve25519.h"
#include "../test_ed25519_cctv.c"

#define MAX_CNT (256UL)
#define MAX_MSG (1232UL)

static uchar prv [ MAX_CNT ][ 32 ];
static uchar pub [ MAX_CNT ][ 32 ];
static uchar sig [ MAX_CNT ][ 64 ];
static uchar msg [ MAX_CNT ][ MAX_MSG ];

static uchar const * pub_p[ MAX_CNT ];
static uchar const * sig_p[ MAX_CNT ];
static uchar const * msg_p[ MAX_CNT ];
static ulong         msg_z[ MAX_CNT ];
static int           ok   [ MAX_CNT ];

static void
gen( fd_rng_t *    rng,
     fd_sha512_t * sha,
     ulong         cnt,
     ulong         msg_sz ) {
  for( ulong i=0UL; i<cnt; i++ ) {
    for( int b=0; b<32; b++ ) prv[i][b] = fd_rng_uchar( rng );
    fd_ed25519_public_from_private( pub[i], prv[i], sha );
    for( ulong b=0UL; b<msg_sz; b++ ) msg[i][b] = fd_rng_uchar( rng );
    fd_ed25519_sign( sig[i], msg[i], msg_sz, pub[i], prv[i], sha );
    pub_p[i] = pub[i];
    sig_p[i] = sig[i];
    msg_p[i] = msg[i];
    msg_z[i] = msg_sz;
  }
}

/* differential compares the x8 verdicts against fd_ed25519_verify's, one
   signature at a time. */

static void
differential( fd_sha512_t * sha,
              ulong         cnt,
              char const *  what ) {
  fd_ed25519_verify_batch_x8( msg_p, msg_z, sig_p, pub_p, ok, cnt );
  for( ulong i=0UL; i<cnt; i++ ) {
    int want = fd_ed25519_verify( msg_p[i], msg_z[i], sig_p[i], pub_p[i], sha );
    if( FD_LIKELY( ok[i]==want ) ) continue;

    /* The one documented classification difference: the x8 path never
       decompresses R, so an undecodable R surfaces as ERR_MSG rather than
       ERR_SIG.  Insist that R really is undecodable -- a genuine verdict
       disagreement must still fail the test. */
    if( want==FD_ED25519_ERR_SIG && ok[i]==FD_ED25519_ERR_MSG &&
        !fd_ed25519_point_validate( sig_p[i] ) ) continue;

    FD_LOG_ERR(( "%s: signature %lu: x8 gave %i, fd_ed25519_verify gave %i", what, i, ok[i], want ));
  }
}

/* Encodings that decode to one of the eight small order points.  Both
   implementations must reject every one of them, in either position. */

static uchar const small_order[][32] = {
  { 0x01,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0 },
  { 0x00,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0 },
  { 0x00,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0x80 },
  { 0xec,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x7f },
  { 0xed,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x7f },
  { 0xee,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x7f },
  { 0x26,0xe8,0x95,0x8f,0xc2,0xb2,0x27,0xb0, 0x45,0xc3,0xf4,0x89,0xf2,0xef,0x98,0xf0,
    0xd5,0xdf,0xac,0x05,0xd3,0xc6,0x33,0x39, 0xb1,0x38,0x02,0x88,0x6d,0x53,0xfc,0x05 },
  { 0xc7,0x17,0x6a,0x70,0x3d,0x4d,0xd8,0x4f, 0xba,0x3c,0x0b,0x76,0x0d,0x10,0x67,0x0f,
    0x2a,0x20,0x53,0xfa,0x2c,0x39,0xcc,0xc6, 0x4e,0xc7,0xfd,0x77,0x92,0xac,0x03,0x7a },
};

/* cctv runs the whole C2SP "Taming the many EdDSAs" corpus through the batch
   path and compares against the corpus's own recorded verdict, which is
   fd_ed25519_verify's.  This is the corpus that exercises non-canonical
   encodings and every small-order point, so it is the one that would catch a
   batch path that had drifted from Firedancer's acceptance predicate.

   Vectors are fed in mixed-size groups rather than one at a time so that a
   rejected lane sits next to accepted ones. */

static void
cctv( void ) {
  ulong n = sizeof(ed25519_verify_cctvs)/sizeof(ed25519_verify_cctvs[0]);

  static uchar const * c_msg[ MAX_CNT ];
  static ulong         c_sz [ MAX_CNT ];
  static uchar const * c_sig[ MAX_CNT ];
  static uchar const * c_pub[ MAX_CNT ];
  static int           c_ok [ MAX_CNT ];

  ulong checked = 0UL;
  for( ulong base=0UL; base<n; base+=17UL ) { /* 17 is coprime with 8 */
    ulong cnt = fd_ulong_min( n-base, 17UL );
    for( ulong i=0UL; i<cnt; i++ ) {
      fd_ed25519_verify_cctv_t const * v = &ed25519_verify_cctvs[ base+i ];
      c_msg[i] = v->msg;
      c_sz [i] = v->msg_sz;
      c_sig[i] = v->sig;
      c_pub[i] = v->pub;
    }
    fd_ed25519_verify_batch_x8( c_msg, c_sz, c_sig, c_pub, c_ok, cnt );
    for( ulong i=0UL; i<cnt; i++ ) {
      fd_ed25519_verify_cctv_t const * v = &ed25519_verify_cctvs[ base+i ];
      int got  = c_ok[i]==FD_ED25519_SUCCESS;
      int want = v->ok;
      if( FD_UNLIKELY( got!=want ) )
        FD_LOG_ERR(( "cctv tc_id=%u (%s): x8 %s, corpus says %s",
                     v->tc_id, v->comment,
                     got ? "accepted" : "rejected", want ? "accept" : "reject" ));
      checked++;
    }
  }
  FD_LOG_NOTICE(( "CCTV corpus: %lu/%lu vectors match fd_ed25519_verify", checked, n ));
}

int
main( int     argc,
      char ** argv ) {
  fd_boot( &argc, &argv );
  int bench = fd_env_strip_cmdline_contains( &argc, &argv, "--bench" );

  fd_rng_t _rng[1]; fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, 0U, 0UL ) );
  fd_sha512_t _sha[1]; fd_sha512_t * sha = fd_sha512_join( fd_sha512_new( _sha ) );

  /* Every batch width from 1 through 24 exercises the tail handling, then
     a couple of widths past the 64 signature inversion chunk. */
  for( ulong cnt=1UL; cnt<=24UL; cnt++ ) {
    gen( rng, sha, cnt, 200UL );
    differential( sha, cnt, "valid" );
  }
  for( ulong cnt=63UL; cnt<=66UL; cnt++ ) {
    gen( rng, sha, cnt, 1232UL );
    differential( sha, cnt, "valid-chunked" );
  }
  gen( rng, sha, 129UL, 0UL );
  differential( sha, 129UL, "valid-empty-msg" );
  FD_LOG_NOTICE(( "valid signatures: ok" ));

  /* Corrupted messages, signatures and public keys. */
  gen( rng, sha, 64UL, 200UL );
  for( ulong i=0UL; i<64UL; i+=3UL ) msg[i][ fd_rng_ulong( rng )%200UL ] ^= (uchar)(1u<<(fd_rng_uint(rng)&7u));
  for( ulong i=1UL; i<64UL; i+=5UL ) sig[i][ fd_rng_ulong( rng )%64UL  ] ^= (uchar)(1u<<(fd_rng_uint(rng)&7u));
  for( ulong i=2UL; i<64UL; i+=7UL ) pub[i][ fd_rng_ulong( rng )%32UL  ] ^= (uchar)(1u<<(fd_rng_uint(rng)&7u));
  differential( sha, 64UL, "corrupted" );
  FD_LOG_NOTICE(( "corrupted inputs: ok" ));

  /* Small order A and small order R, mixed with valid signatures so that a
     rejected lane cannot disturb its neighbours. */
  ulong n_small = sizeof(small_order)/sizeof(small_order[0]);
  gen( rng, sha, 32UL, 200UL );
  for( ulong i=0UL; i<n_small; i++ ) {
    fd_memcpy( pub[ 2*i     ], small_order[i], 32UL );
    fd_memcpy( sig[ 2*i + 1 ], small_order[i], 32UL );
  }
  differential( sha, 32UL, "small-order" );
  FD_LOG_NOTICE(( "small order rejection: ok" ));

  /* Non canonical S (S >= L) must be rejected by both. */
  gen( rng, sha, 8UL, 200UL );
  for( int b=0; b<32; b++ ) sig[3][32+b] = 0xff;
  sig[3][63] = 0x1f;
  differential( sha, 8UL, "non-canonical-S" );
  FD_LOG_NOTICE(( "non canonical S rejection: ok" ));

  /* Public keys that are not on the curve at all. */
  gen( rng, sha, 16UL, 200UL );
  for( ulong i=0UL; i<16UL; i+=4UL ) { pub[i][31] &= 0x7f; pub[i][0] ^= 0x11; }
  differential( sha, 16UL, "bad-pubkey" );
  FD_LOG_NOTICE(( "invalid public key rejection: ok" ));

  cctv();

  if( bench ) {
    static ulong const sizes[] = { 200UL, 1232UL };
    static ulong const counts[] = { 1UL, 2UL, 4UL, 8UL, 16UL, 64UL, 128UL };

    for( ulong si=0UL; si<2UL; si++ ) {
      ulong msg_sz = sizes[si];
      gen( rng, sha, 128UL, msg_sz );

      /* Baseline: fd_ed25519_verify in a loop, same fixture. */
      {
        ulong iter = 20000UL;
        int   sink = 0;
        long dt = -fd_log_wallclock();
        for( ulong r=0UL; r<iter; r++ ) {
          sink += fd_ed25519_verify( msg_p[r&127UL], msg_sz, sig_p[r&127UL], pub_p[r&127UL], sha );
          FD_COMPILER_FORGET( sink );
        }
        dt += fd_log_wallclock();
        if( FD_UNLIKELY( sink ) ) FD_LOG_ERR(( "bench baseline rejected a valid fixture" ));
        FD_LOG_NOTICE(( "msg=%4lu fd_ed25519_verify         %8.3f us/sig %10.0f sig/s/core",
                        msg_sz, (double)dt/(double)iter/1e3, 1e9*(double)iter/(double)dt ));
      }

      for( ulong ci=0UL; ci<7UL; ci++ ) {
        ulong cnt  = counts[ci];
        ulong iter = fd_ulong_max( 200000UL/cnt, 20UL );
        /* warm */
        fd_ed25519_verify_batch_x8( msg_p, msg_z, sig_p, pub_p, ok, cnt );
        for( ulong i=0UL; i<cnt; i++ )
          if( FD_UNLIKELY( ok[i]!=FD_ED25519_SUCCESS ) ) FD_LOG_ERR(( "bench fixture rejected" ));
        int  sink = 0;
        long dt = -fd_log_wallclock();
        for( ulong r=0UL; r<iter; r++ ) {
          sink += fd_ed25519_verify_batch_x8( msg_p, msg_z, sig_p, pub_p, ok, cnt );
          FD_COMPILER_FORGET( sink );
        }
        dt += fd_log_wallclock();
        if( FD_UNLIKELY( sink ) ) FD_LOG_ERR(( "bench rejected a valid fixture" ));
        double per = (double)dt/(double)(iter*cnt);
        FD_LOG_NOTICE(( "msg=%4lu verify_batch_x8 n=%-4lu   %8.3f us/sig %10.0f sig/s/core",
                        msg_sz, cnt, per/1e3, 1e9/per ));
      }
    }
  }

  fd_sha512_delete( fd_sha512_leave( sha ) );
  fd_rng_delete( fd_rng_leave( rng ) );
  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}

#endif
