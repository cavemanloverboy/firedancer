#ifndef HEADER_fd_src_ballet_ed25519_avx512_fd_r51x8_h
#define HEADER_fd_src_ballet_ed25519_avx512_fd_r51x8_h

#if FD_HAS_AVX512

#include "../../../util/fd_util_base.h"
#include <immintrin.h>

/* fd_r51x8_t holds EIGHT INDEPENDENT GF(p) elements, p = 2^255-19, one per
   AVX-512 lane.  This is a different vectorization axis from fd_r43x6_t: an
   fd_r43x6_t spreads ONE field element across the lanes of a single vector,
   whereas an fd_r51x8_t packs eight unrelated field elements limb-major:

     limb[i] lane j  ==  the i-th radix-2^51 limb of the j-th element

   so element j is

     ( l0 + l1 2^51 + l2 2^102 + l3 2^153 + l4 2^204 ) mod p

   with ln = limb[n] lane j.

   The tradeoff:

   - fd_r43x6 wins on LATENCY for a single dependent chain.  One field
     multiply is a handful of VPMADD52 with no cross-element parallelism to
     find, which is what fd_ed25519_verify needs when it verifies exactly
     one signature.

   - fd_r51x8 wins on THROUGHPUT when there are >=8 independent signatures
     in flight.  Every VPMADD52 does eight signatures' worth of work, so the
     per-signature cost of a field multiply drops by roughly the lane count
     even though the instruction count per multiply goes up.

   Representation contract, which every operation below both requires and
   restores:

     each limb lane is in [0,2^52)

   That is the VPMADD52 multiplicand bound, NOT a canonical representative:
   an element has many valid fd_r51x8_t encodings, and the value is only
   defined modulo p.  fd_r51x8_tobytes is the only operation that produces a
   unique answer, and it fully reduces.

   Lanes are completely independent.  No operation moves data between lanes,
   and no operation branches on lane content, so a lane holding garbage
   (a padded tail slot, a signature already known to be invalid) can never
   corrupt a sibling lane.  Operations are NOT constant time with respect to
   which lanes are active, but the data they run on -- public keys, signature
   points, challenge scalars -- is public in Ed25519 verification.  Do not
   use this header for signing or key agreement. */

typedef struct { __m512i limb[5]; } fd_r51x8_t;

#define FD_R51X8_LIMB_BITS (51)
#define FD_R51X8_LIMB_MASK ((1UL<<51)-1UL)

/* Two domains, because carry propagation is a real fraction of the cost
   and most of it is avoidable:

     tight -- every limb lane below 2^52.  This is VPMADD52's multiplicand
              bound, so only a tight value may be an input to _mul.

     wide  -- every limb lane below 2^62.  Cheap to produce (no carry
              chain) and closed under addition and subtraction; only
              fd_r51x8_carry turns it back into a tight value.

   The _w suffix means "leaves the result wide".  A chain like
   mulw -> addw -> sub therefore pays ONE carry instead of three.  Every
   function below documents which domain it needs and which it returns;
   feeding a wide value into _mul silently drops bits above bit 52. */

/* Nothing here is force inlined by accident: these are the whole hot loop,
   and letting the compiler keep them out of line costs both the call and,
   much worse, any scheduling overlap between one operation's carry chain
   and the next operation's multiplies. */
#define FD_R51X8_INLINE static inline __attribute__((always_inline))

/* fd_r51x8_bcast broadcasts one scalar radix-2^51 element (5 limbs, each
   below 2^52) into all eight lanes. */

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_bcast( ulong const l[ static 5 ] ) {
  fd_r51x8_t z;
  for( int i=0; i<5; i++ ) z.limb[i] = _mm512_set1_epi64( (long)l[i] );
  return z;
}

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_zero( void ) {
  fd_r51x8_t z;
  for( int i=0; i<5; i++ ) z.limb[i] = _mm512_setzero_si512();
  return z;
}

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_one( void ) {
  fd_r51x8_t z = fd_r51x8_zero();
  z.limb[0] = _mm512_set1_epi64( 1L );
  return z;
}

/* fd_r51x8_carry maps wide (< 2^62) back to tight (< 2^52).

   All five carries are taken from the ORIGINAL limbs before any of them is
   updated, so the five lanes of work are independent and the compiler can
   schedule them freely instead of serializing a five step carry chain.
   Each carry is then below 2^11, which makes limb 0 below 2^51+19*2^11 and
   limbs 1..4 below 2^51+2^11 -- all far below 2^52.

   The 19*carry lands in VPMADD52's input range (the carry is tiny), so it
   is one IFMA rather than the shift-shift-add-add that 19*x needs when x
   can be large. */

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_carry( fd_r51x8_t x ) {
  __m512i const mask = _mm512_set1_epi64( (long)FD_R51X8_LIMB_MASK );
  __m512i const k19  = _mm512_set1_epi64( 19L );
  __m512i c[5], l[5];
  for( int i=0; i<5; i++ ) {
    c[i] = _mm512_srli_epi64( x.limb[i], FD_R51X8_LIMB_BITS );
    l[i] = _mm512_and_si512 ( x.limb[i], mask );
  }
  fd_r51x8_t z;
  z.limb[0] = _mm512_madd52lo_epu64( l[0], c[4], k19 );
  z.limb[1] = _mm512_add_epi64( l[1], c[0] );
  z.limb[2] = _mm512_add_epi64( l[2], c[1] );
  z.limb[3] = _mm512_add_epi64( l[3], c[2] );
  z.limb[4] = _mm512_add_epi64( l[4], c[3] );
  return z;
}

/* addw/subw: wide in, wide out, no carry.

   subw adds 1024*p first so no lane can underflow.  1024*p has limbs just
   under 2^61, which covers any wide operand, and the sum stays below 2^63
   so the unsigned subtract is exact. */

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_addw( fd_r51x8_t x, fd_r51x8_t y ) {
  fd_r51x8_t z;
  for( int i=0; i<5; i++ ) z.limb[i] = _mm512_add_epi64( x.limb[i], y.limb[i] );
  return z;
}

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_subw( fd_r51x8_t x, fd_r51x8_t y ) {
  __m512i const bias0 = _mm512_set1_epi64( (long)(1024UL*((1UL<<51)-19UL)) );
  __m512i const biasn = _mm512_set1_epi64( (long)(1024UL*((1UL<<51)- 1UL)) );
  fd_r51x8_t z;
  z.limb[0] = _mm512_sub_epi64( _mm512_add_epi64( x.limb[0], bias0 ), y.limb[0] );
  for( int i=1; i<5; i++ )
    z.limb[i] = _mm512_sub_epi64( _mm512_add_epi64( x.limb[i], biasn ), y.limb[i] );
  return z;
}

FD_R51X8_INLINE fd_r51x8_t fd_r51x8_add( fd_r51x8_t x, fd_r51x8_t y ) { return fd_r51x8_carry( fd_r51x8_addw( x, y ) ); }
FD_R51X8_INLINE fd_r51x8_t fd_r51x8_sub( fd_r51x8_t x, fd_r51x8_t y ) { return fd_r51x8_carry( fd_r51x8_subw( x, y ) ); }

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_neg( fd_r51x8_t x ) {
  return fd_r51x8_sub( fd_r51x8_zero(), x );
}

/* fd_r51x8_mulw is the schoolbook product, accumulated ONE OUTPUT COLUMN AT
   A TIME.

   The obvious arrangement -- 9 low and 9 high accumulators alive across all
   25 partial products -- needs 28 live vectors once the operands are
   counted, which is over budget on a 32 register machine and costs a pile
   of register-to-register moves.  Accumulating per output column instead
   keeps four accumulators alive, and the five columns are independent, so
   there is plenty for the scheduler to overlap.

   The 19-fold is applied to accumulated columns rather than to a
   multiplicand because 19*2^51 exceeds VPMADD52's 52-bit input bound.  The
   high half of a product splits at bit 52 while the limb weight is 2^51, so
   it lands in the next column with weight 2 -- hence the shifts.

   Bounds: a column sums at most five 52-bit values, so
   a < 2^54.4 + 2^55.4 < 2^55.6, b likewise, and 19*b < 2^60.2.  That is
   inside fd_r51x8_carry's wide input contract. */

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_mulw( fd_r51x8_t x, fd_r51x8_t y ) {
  __m512i lo[9], hi[9];
  for( int d=0; d<9; d++ ) { lo[d] = _mm512_setzero_si512(); hi[d] = _mm512_setzero_si512(); }

  for( int i=0; i<5; i++ )
    for( int j=0; j<5; j++ ) {
      lo[i+j] = _mm512_madd52lo_epu64( lo[i+j], x.limb[i], y.limb[j] );
      hi[i+j] = _mm512_madd52hi_epu64( hi[i+j], x.limb[i], y.limb[j] );
    }

  /* coef[d] = lo[d] + 2*hi[d-1] */
  __m512i coef[10];
  coef[0] = lo[0];
  for( int d=1; d<9; d++ ) coef[d] = _mm512_add_epi64( lo[d], _mm512_slli_epi64( hi[d-1], 1 ) );
  coef[9] = _mm512_slli_epi64( hi[8], 1 );

  fd_r51x8_t z;
  for( int k=0; k<5; k++ ) {
    __m512i b   = coef[k+5];
    __m512i b19 = _mm512_add_epi64( _mm512_add_epi64( b, _mm512_slli_epi64( b, 1 ) ),
                                    _mm512_slli_epi64( b, 4 ) );
    z.limb[k] = _mm512_add_epi64( coef[k], b19 );
  }
  return z;
}

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_mul( fd_r51x8_t x, fd_r51x8_t y ) {
  return fd_r51x8_carry( fd_r51x8_mulw( x, y ) );
}

/* fd_r51x8_sqrw halves the partial product count: x_i*x_j and x_j*x_i are
   the same, so only i<=j is issued and the off-diagonal terms are doubled
   afterwards.  30 VPMADD52 instead of 50.

   Doubling has to happen on the ACCUMULATOR, not on a multiplicand, since
   2*x_j would leave the 52-bit input range.  Written out, one column is

     a = dlo + 2*( clo + dhi' + 2*chi' )

   where d/c are the diagonal and cross accumulators and the primed ones
   belong to the previous column -- which folds the high-half doubling and
   the cross doubling into the same two shifts.

   VPMADD52 throughput is what this loop is limited by on Zen 4/Zen 5, so
   trading 20 IFMA for ~8 extra shifts and adds per column is a clear win
   even though the total instruction count barely moves. */

FD_R51X8_INLINE fd_r51x8_t
fd_r51x8_sqrw( fd_r51x8_t x ) {
  __m512i dlo[9], dhi[9], clo[9], chi[9];
  for( int d=0; d<9; d++ ) {
    dlo[d] = _mm512_setzero_si512(); dhi[d] = _mm512_setzero_si512();
    clo[d] = _mm512_setzero_si512(); chi[d] = _mm512_setzero_si512();
  }

  for( int i=0; i<5; i++ ) {
    dlo[2*i] = _mm512_madd52lo_epu64( dlo[2*i], x.limb[i], x.limb[i] );
    dhi[2*i] = _mm512_madd52hi_epu64( dhi[2*i], x.limb[i], x.limb[i] );
    for( int j=i+1; j<5; j++ ) {
      clo[i+j] = _mm512_madd52lo_epu64( clo[i+j], x.limb[i], x.limb[j] );
      chi[i+j] = _mm512_madd52hi_epu64( chi[i+j], x.limb[i], x.limb[j] );
    }
  }

  __m512i lo[9], hi[9];
  for( int d=0; d<9; d++ ) {
    lo[d] = _mm512_add_epi64( dlo[d], _mm512_slli_epi64( clo[d], 1 ) );
    hi[d] = _mm512_add_epi64( dhi[d], _mm512_slli_epi64( chi[d], 1 ) );
  }

  __m512i coef[10];
  coef[0] = lo[0];
  for( int d=1; d<9; d++ ) coef[d] = _mm512_add_epi64( lo[d], _mm512_slli_epi64( hi[d-1], 1 ) );
  coef[9] = _mm512_slli_epi64( hi[8], 1 );

  fd_r51x8_t z;
  for( int k=0; k<5; k++ ) {
    __m512i b   = coef[k+5];
    __m512i b19 = _mm512_add_epi64( _mm512_add_epi64( b, _mm512_slli_epi64( b, 1 ) ),
                                    _mm512_slli_epi64( b, 4 ) );
    z.limb[k] = _mm512_add_epi64( coef[k], b19 );
  }
  return z;
}

FD_R51X8_INLINE fd_r51x8_t fd_r51x8_sqr( fd_r51x8_t x ) { return fd_r51x8_carry( fd_r51x8_sqrw( x ) ); }

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_sqr_n( fd_r51x8_t x, int n ) {
  for( int i=0; i<n; i++ ) x = fd_r51x8_sqr( x );
  return x;
}

/* fd_r51x8_if selects y where the corresponding mask bit is set and x
   elsewhere. */

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_if( __mmask8 c, fd_r51x8_t y, fd_r51x8_t x ) {
  fd_r51x8_t z;
  for( int i=0; i<5; i++ ) z.limb[i] = _mm512_mask_blend_epi64( c, x.limb[i], y.limb[i] );
  return z;
}

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_neg_if( __mmask8 c, fd_r51x8_t x ) {
  return fd_r51x8_if( c, fd_r51x8_neg( x ), x );
}

/* fd_r51x8_reduce produces the unique representative in [0,p) in every
   lane.  It is used only at the boundaries (byte export, equality), never
   inside the point loops. */

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_reduce( fd_r51x8_t x ) {
  __m512i const mask = _mm512_set1_epi64( (long)FD_R51X8_LIMB_MASK );

  /* Two sequential carry chains bring every limb below 2^51 with limb 0
     below 2^51+19 -- i.e. the value below 2^255+something small. */
  for( int pass=0; pass<2; pass++ ) {
    __m512i c = _mm512_setzero_si512();
    for( int i=0; i<5; i++ ) {
      __m512i t = _mm512_add_epi64( x.limb[i], c );
      c = _mm512_srli_epi64( t, FD_R51X8_LIMB_BITS );
      x.limb[i] = _mm512_and_si512( t, mask );
    }
    x.limb[0] = _mm512_madd52lo_epu64( x.limb[0], c, _mm512_set1_epi64( 19L ) );
  }

  /* Now the value is below 2^255+19*small.  Conditionally subtract p, twice
     is unnecessary: after the loop limb0 < 2^51+19*19, so value < 2p. */
  __m512i q = _mm512_srli_epi64( _mm512_add_epi64( x.limb[0], _mm512_set1_epi64( 19L ) ),
                                 FD_R51X8_LIMB_BITS );
  for( int i=1; i<5; i++ )
    q = _mm512_srli_epi64( _mm512_add_epi64( x.limb[i], q ), FD_R51X8_LIMB_BITS );

  /* value - p*q, q in {0,1} */
  x.limb[0] = _mm512_add_epi64( x.limb[0], _mm512_mullo_epi64( q, _mm512_set1_epi64( 19L ) ) );
  __m512i c = _mm512_setzero_si512();
  for( int i=0; i<5; i++ ) {
    __m512i t = _mm512_add_epi64( x.limb[i], c );
    c = _mm512_srli_epi64( t, FD_R51X8_LIMB_BITS );
    x.limb[i] = _mm512_and_si512( t, mask );
  }
  /* Drop the 2^255 overflow bit: limb 4's bit 51 was already masked off. */
  return x;
}

/* fd_r51x8_is_zero returns a mask of lanes whose value is zero mod p. */

FD_FN_UNUSED static inline __mmask8
fd_r51x8_is_zero( fd_r51x8_t x ) {
  fd_r51x8_t r = fd_r51x8_reduce( x );
  __m512i acc = r.limb[0];
  for( int i=1; i<5; i++ ) acc = _mm512_or_si512( acc, r.limb[i] );
  return _mm512_cmpeq_epi64_mask( acc, _mm512_setzero_si512() );
}

FD_FN_UNUSED static inline __mmask8
fd_r51x8_eq( fd_r51x8_t x, fd_r51x8_t y ) {
  return fd_r51x8_is_zero( fd_r51x8_sub( x, y ) );
}

/* fd_r51x8_sgn returns a mask of lanes whose canonical representative is
   odd -- the Ed25519 compressed-point sign bit. */

FD_FN_UNUSED static inline __mmask8
fd_r51x8_sgn( fd_r51x8_t x ) {
  fd_r51x8_t r = fd_r51x8_reduce( x );
  return _mm512_test_epi64_mask( r.limb[0], _mm512_set1_epi64( 1L ) );
}

/* fd_r51x8_pow22523 computes x^((p-5)/8), the exponentiation behind both
   the inverse and the square root of a ratio.  The addition chain is the
   standard ref10 one; every step runs all eight lanes at once. */

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_pow22523( fd_r51x8_t z ) {
  fd_r51x8_t t0, t1, t2;
  t0 = fd_r51x8_sqr  ( z );                     /* z^2      */
  t1 = fd_r51x8_sqr_n( t0, 2 );                 /* z^8      */
  t1 = fd_r51x8_mul  ( z, t1 );                 /* z^9      */
  t0 = fd_r51x8_mul  ( t0, t1 );                /* z^11     */
  t0 = fd_r51x8_sqr  ( t0 );                    /* z^22     */
  t0 = fd_r51x8_mul  ( t1, t0 );                /* z^31     = z^(2^5-1)  */
  t1 = fd_r51x8_sqr_n( t0, 5  ); t0 = fd_r51x8_mul( t1, t0 ); /* 2^10-1 */
  t1 = fd_r51x8_sqr_n( t0, 10 ); t1 = fd_r51x8_mul( t1, t0 ); /* 2^20-1 */
  t2 = fd_r51x8_sqr_n( t1, 20 ); t1 = fd_r51x8_mul( t2, t1 ); /* 2^40-1 */
  t1 = fd_r51x8_sqr_n( t1, 10 ); t0 = fd_r51x8_mul( t1, t0 ); /* 2^50-1 */
  t1 = fd_r51x8_sqr_n( t0, 50 ); t1 = fd_r51x8_mul( t1, t0 ); /* 2^100-1 */
  t2 = fd_r51x8_sqr_n( t1,100 ); t1 = fd_r51x8_mul( t2, t1 ); /* 2^200-1 */
  t1 = fd_r51x8_sqr_n( t1, 50 ); t0 = fd_r51x8_mul( t1, t0 ); /* 2^250-1 */
  t0 = fd_r51x8_sqr_n( t0,  2 );
  return fd_r51x8_mul( t0, z );                 /* z^((p-5)/8) */
}

/* fd_r51x8_inv computes x^(p-2).  Only used to close a Montgomery batch
   inversion, so its cost is amortized over the whole batch. */

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_inv( fd_r51x8_t z ) {
  fd_r51x8_t t  = fd_r51x8_pow22523( z );                    /* z^(2^252-3) */
  t = fd_r51x8_sqr_n( t, 3 );                                /* z^(p-5)     */
  fd_r51x8_t z3 = fd_r51x8_mul( fd_r51x8_sqr( z ), z );      /* z^3         */
  return fd_r51x8_mul( t, z3 );                              /* z^(p-2)     */
}

/* fd_r51x8_st / fd_r51x8_ld move between the vector form and a plain
   [limb][lane] array.  Going through memory rather than casting a __m512i
   to ulong* keeps the compiler's aliasing analysis honest, and the store
   and reload are exactly what the lane-at-a-time transposes below need. */

FD_FN_UNUSED static inline void
fd_r51x8_st( ulong      out[ 5 ][ 8 ],
             fd_r51x8_t x ) {
  for( int i=0; i<5; i++ ) _mm512_storeu_si512( (void *)out[i], x.limb[i] );
}

FD_FN_UNUSED static inline fd_r51x8_t
fd_r51x8_ld( ulong const in[ 5 ][ 8 ] ) {
  fd_r51x8_t z;
  for( int i=0; i<5; i++ ) z.limb[i] = _mm512_loadu_si512( (void const *)in[i] );
  return z;
}

/* fd_r51x8_unpack expands one 32-byte little endian value into radix-2^51
   limbs.  The high bit is ignored (it is the sign bit in a compressed
   Edwards point) and NO canonicity check is performed: a y in [p,2^255)
   simply produces a non-canonical but arithmetically valid representative,
   which is what fd_f25519_frombytes does too. */

FD_FN_UNUSED static inline void
fd_r51x8_unpack( ulong       l [ static 5 ],
                 uchar const in[ static 32 ] ) {
  ulong w0, w1, w2, w3;
  memcpy( &w0, in,     8 );
  memcpy( &w1, in+ 8,  8 );
  memcpy( &w2, in+16,  8 );
  memcpy( &w3, in+24,  8 );
  w3 &= 0x7fffffffffffffffUL;

  l[0] =   w0                  & FD_R51X8_LIMB_MASK;
  l[1] = ((w0>>51) | (w1<<13)) & FD_R51X8_LIMB_MASK;
  l[2] = ((w1>>38) | (w2<<26)) & FD_R51X8_LIMB_MASK;
  l[3] = ((w2>>25) | (w3<<39)) & FD_R51X8_LIMB_MASK;
  l[4] =  (w3>>12)             & FD_R51X8_LIMB_MASK;
}

/* fd_r51x8_pack exports one canonical 32-byte little endian value.  The
   limbs must already be reduced. */

FD_FN_UNUSED static inline void
fd_r51x8_pack( uchar       out[ static 32 ],
               ulong const l  [ static 5  ] ) {
  ulong w0 =  l[0]      | (l[1]<<51);
  ulong w1 = (l[1]>>13) | (l[2]<<38);
  ulong w2 = (l[2]>>26) | (l[3]<<25);
  ulong w3 = (l[3]>>39) | (l[4]<<12);

  memcpy( out,     &w0, 8 );
  memcpy( out+ 8,  &w1, 8 );
  memcpy( out+16,  &w2, 8 );
  memcpy( out+24,  &w3, 8 );
}

#endif /* FD_HAS_AVX512 */
#endif /* HEADER_fd_src_ballet_ed25519_avx512_fd_r51x8_h */
