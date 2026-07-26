#include "fd_ed25519_x8.h"

#if FD_HAS_AVX512

#include "fd_r51x8.h"
#include "../fd_curve25519_scalar.h"
#include "../../sha512/fd_sha512.h"
#include <string.h>

/**********************************************************************/
/* Curve constants                                                    */
/**********************************************************************/

/* d = -121665/121666, d2 = 2*d and sqrt(-1), in radix 2^51.  These are
   checked against the curve equation at table build time (see
   fd_ed25519_x8_selftest) so a transcription error cannot go unnoticed. */

static ulong const fd_ed25519_x8_d[5] = {
  929955233495203UL, 466365720129213UL, 1662059464998953UL,
  2033849074728123UL, 1442794654840575UL
};

static ulong const fd_ed25519_x8_d2[5] = {
  1859910466990425UL, 932731440258426UL, 1072319116312658UL,
  1815898335770999UL, 633789495995903UL
};

static ulong const fd_ed25519_x8_sqrtm1[5] = {
  1718705420411056UL, 234908883556509UL, 2233514472574048UL,
  2117202627021982UL, 765476049583133UL
};

/* The Ed25519 base point, as its standard compressed encoding.  Decoding
   it with the same routine the verifier uses avoids hand transcribing two
   more 5-limb constants. */

static uchar const fd_ed25519_x8_base_enc[32] = {
  0x58,0x66,0x66,0x66,0x66,0x66,0x66,0x66, 0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,
  0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66, 0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66
};

/**********************************************************************/
/* Points                                                             */
/**********************************************************************/

/* Extended twisted Edwards coordinates, eight independent points. */

typedef struct { fd_r51x8_t X, Y, Z, T; } fd_ed25519_x8_t;

/* ge_cached: (Y+X, Y-X, Z, 2d*T) for a general point. */

typedef struct { fd_r51x8_t ypx, ymx, z, t2d; } fd_ed25519_x8_cached_t;

/* ge_precomp: (y+x, y-x, 2d*x*y) for an affine point, i.e. Z == 1. */

typedef struct { fd_r51x8_t ypx, ymx, t2d; } fd_ed25519_x8_precomp_t;

/* One affine precomputed point, stored once rather than once per lane.
   The fixed base comb table is shared by every lane and every thread, so
   keeping it scalar keeps it in L2: 64 positions x 8 magnitudes x 120
   bytes is 61 KiB. */

typedef struct { ulong ypx[5], ymx[5], t2d[5]; } fd_ed25519_x8_comb_entry_t;

static fd_ed25519_x8_comb_entry_t fd_ed25519_x8_comb[64][8] __attribute__((aligned(64)));

static inline fd_ed25519_x8_t
fd_ed25519_x8_identity( void ) {
  fd_ed25519_x8_t p;
  p.X = fd_r51x8_zero();
  p.Y = fd_r51x8_one();
  p.Z = fd_r51x8_one();
  p.T = fd_r51x8_zero();
  return p;
}

/* fd_ed25519_x8_dbl is ref10's ge_p2_dbl followed by ge_p1p1_to_p3.  T is
   only consumed by an addition, so callers in the middle of a doubling run
   can drop it -- see want_t. */

static inline fd_ed25519_x8_t
fd_ed25519_x8_dbl( fd_ed25519_x8_t const * p,
                   int                     want_t ) {
  /* The squares stay wide: their only consumers are additions, which are
     closed over the wide domain.  Only the values that feed a multiply are
     carried back down. */
  fd_r51x8_t xx = fd_r51x8_sqrw( p->X );
  fd_r51x8_t yy = fd_r51x8_sqrw( p->Y );
  fd_r51x8_t zz = fd_r51x8_sqrw( p->Z );
  fd_r51x8_t b  = fd_r51x8_addw( zz, zz );
  fd_r51x8_t a  = fd_r51x8_add ( p->X, p->Y );  /* multiply input, carry */
  fd_r51x8_t aa = fd_r51x8_sqrw( a );

  fd_r51x8_t y3 = fd_r51x8_add( yy, xx );
  fd_r51x8_t z3 = fd_r51x8_sub( yy, xx );
  fd_r51x8_t x3 = fd_r51x8_sub( aa, y3 );
  fd_r51x8_t t3 = fd_r51x8_sub( b,  z3 );

  fd_ed25519_x8_t r;
  r.X = fd_r51x8_mul( x3, t3 );
  r.Y = fd_r51x8_mul( y3, z3 );
  r.Z = fd_r51x8_mul( z3, t3 );
  r.T = want_t ? fd_r51x8_mul( x3, y3 ) : fd_r51x8_zero();
  return r;
}

/* fd_ed25519_x8_add is ref10's ge_add followed by ge_p1p1_to_p3. */

static inline fd_ed25519_x8_t
fd_ed25519_x8_add( fd_ed25519_x8_t const *        p,
                   fd_ed25519_x8_cached_t const * q ) {
  fd_r51x8_t a  = fd_r51x8_mulw( fd_r51x8_add( p->Y, p->X ), q->ypx );
  fd_r51x8_t b  = fd_r51x8_mulw( fd_r51x8_sub( p->Y, p->X ), q->ymx );
  fd_r51x8_t c  = fd_r51x8_mulw( q->t2d, p->T );
  fd_r51x8_t d  = fd_r51x8_mulw( p->Z,   q->z );
  d = fd_r51x8_addw( d, d );

  fd_r51x8_t x3 = fd_r51x8_sub( a, b );
  fd_r51x8_t y3 = fd_r51x8_add( a, b );
  fd_r51x8_t z3 = fd_r51x8_add( d, c );
  fd_r51x8_t t3 = fd_r51x8_sub( d, c );

  fd_ed25519_x8_t r;
  r.X = fd_r51x8_mul( x3, t3 );
  r.Y = fd_r51x8_mul( y3, z3 );
  r.Z = fd_r51x8_mul( z3, t3 );
  r.T = fd_r51x8_mul( x3, y3 );
  return r;
}

/* fd_ed25519_x8_madd is ref10's ge_madd: the same addition with Zq == 1,
   which saves one multiply. */

static inline fd_ed25519_x8_t
fd_ed25519_x8_madd( fd_ed25519_x8_t const *         p,
                    fd_ed25519_x8_precomp_t const * q ) {
  fd_r51x8_t a  = fd_r51x8_mulw( fd_r51x8_add( p->Y, p->X ), q->ypx );
  fd_r51x8_t b  = fd_r51x8_mulw( fd_r51x8_sub( p->Y, p->X ), q->ymx );
  fd_r51x8_t c  = fd_r51x8_mulw( q->t2d, p->T );
  fd_r51x8_t d  = fd_r51x8_addw( p->Z, p->Z );

  fd_r51x8_t x3 = fd_r51x8_sub( a, b );
  fd_r51x8_t y3 = fd_r51x8_add( a, b );
  fd_r51x8_t z3 = fd_r51x8_add( d, c );
  fd_r51x8_t t3 = fd_r51x8_sub( d, c );

  fd_ed25519_x8_t r;
  r.X = fd_r51x8_mul( x3, t3 );
  r.Y = fd_r51x8_mul( y3, z3 );
  r.Z = fd_r51x8_mul( z3, t3 );
  r.T = fd_r51x8_mul( x3, y3 );
  return r;
}

static inline fd_ed25519_x8_cached_t
fd_ed25519_x8_to_cached( fd_ed25519_x8_t const * p ) {
  fd_ed25519_x8_cached_t c;
  c.ypx = fd_r51x8_add( p->Y, p->X );
  c.ymx = fd_r51x8_sub( p->Y, p->X );
  c.z   = p->Z;
  c.t2d = fd_r51x8_mul( p->T, fd_r51x8_bcast( fd_ed25519_x8_d2 ) );
  return c;
}

/**********************************************************************/
/* Decompression                                                      */
/**********************************************************************/

/* fd_ed25519_x8_decode decodes eight compressed points at once.  enc[j] is
   lane j's 32 bytes; a NULL entry marks an inactive lane, which decodes to
   the identity so it cannot produce a spurious fault.  Returns the mask of
   lanes that decoded successfully.

   This is the standard ref10 recover-x, except that the single
   exponentiation is shared by all eight lanes: one x^((p-5)/8) chain of
   250 squarings and 11 multiplies now covers eight signatures. */

static __mmask8
fd_ed25519_x8_decode( fd_ed25519_x8_t *   out,
                      uchar const * const enc[ static 8 ],
                      __mmask8            active ) {
  static uchar const zero_enc[32] = {
    1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
  };

  ulong ylimb[5][8];
  __mmask8 sign = (__mmask8)0;
  for( int lane=0; lane<8; lane++ ) {
    uchar const * b = ( (active>>lane)&1 ) ? enc[ lane ] : zero_enc;
    ulong l[5];
    fd_r51x8_unpack( l, b );
    for( int i=0; i<5; i++ ) ylimb[i][lane] = l[i];
    if( b[31] & 0x80 ) sign = (__mmask8)( sign | (1<<lane) );
  }
  sign &= active;
  fd_r51x8_t y = fd_r51x8_ld( (ulong const (*)[8])ylimb );

  fd_r51x8_t one = fd_r51x8_one();
  fd_r51x8_t yy  = fd_r51x8_sqr( y );
  fd_r51x8_t u   = fd_r51x8_sub( yy, one );                                    /* y^2 - 1     */
  fd_r51x8_t v   = fd_r51x8_add( fd_r51x8_mul( yy, fd_r51x8_bcast( fd_ed25519_x8_d ) ), one ); /* d y^2 + 1 */

  fd_r51x8_t v3  = fd_r51x8_mul( fd_r51x8_sqr( v ), v );                       /* v^3         */
  fd_r51x8_t x   = fd_r51x8_mul( fd_r51x8_mul( fd_r51x8_sqr( v3 ), v ), u );   /* u v^7       */
  x = fd_r51x8_pow22523( x );                                                  /* (u v^7)^((p-5)/8) */
  x = fd_r51x8_mul( fd_r51x8_mul( x, v3 ), u );                                /* u v^3 (...) */

  fd_r51x8_t vxx   = fd_r51x8_mul( fd_r51x8_sqr( x ), v );
  __mmask8   good  = fd_r51x8_eq( vxx, u );
  __mmask8   flip  = fd_r51x8_eq( vxx, fd_r51x8_neg( u ) );

  fd_r51x8_t xalt = fd_r51x8_mul( x, fd_r51x8_bcast( fd_ed25519_x8_sqrtm1 ) );
  x = fd_r51x8_if( flip, xalt, x );

  __mmask8 valid = (__mmask8)( ( good | flip ) & active );

  /* x == 0 with the sign bit set has no valid preimage. */
  __mmask8 xzero = fd_r51x8_is_zero( x );
  valid = (__mmask8)( valid & ~( xzero & sign ) );

  /* Choose the root whose parity matches the encoded sign bit. */
  x = fd_r51x8_neg_if( (__mmask8)( sign ^ fd_r51x8_sgn( x ) ), x );

  /* Inactive or failed lanes become the identity: harmless, and keeps Z
     nonzero so the shared batch inversion stays well defined. */
  out->X = fd_r51x8_if( valid, x,   fd_r51x8_zero() );
  out->Y = fd_r51x8_if( valid, y,   fd_r51x8_one()  );
  out->Z = fd_r51x8_one();
  out->T = fd_r51x8_mul( out->X, out->Y );
  return valid;
}

/**********************************************************************/
/* Fixed base comb table                                              */
/**********************************************************************/

/* One extended point in scalar [coordinate][limb] form, used only while
   building the fixed base table. */

typedef struct { ulong c[4][5]; } fd_ed25519_x8_scalar_pt_t;

static inline fd_ed25519_x8_scalar_pt_t
fd_ed25519_x8_get_lane( fd_ed25519_x8_t const * p,
                        int                     lane ) {
  fd_r51x8_t const * src[4] = { &p->X, &p->Y, &p->Z, &p->T };
  fd_ed25519_x8_scalar_pt_t out;
  for( int c=0; c<4; c++ ) {
    ulong tmp[5][8];
    fd_r51x8_st( tmp, *src[c] );
    for( int i=0; i<5; i++ ) out.c[c][i] = tmp[i][lane];
  }
  return out;
}

/* fd_ed25519_x8_build_comb fills fd_ed25519_x8_comb with
   comb[i][j] = [(j+1) * 16^i] B in affine precomputed form, so that a
   radix-16 signed digit expansion of s evaluates [s]B with 64 mixed
   additions and NO doublings.

   Each position's eight multiples are packed one per lane, so the whole
   table costs 64 batch inversions rather than 512 scalar ones. */

static void
fd_ed25519_x8_build_comb( void ) {
  fd_ed25519_x8_t base;
  uchar const * enc[8];
  for( int lane=0; lane<8; lane++ ) enc[ lane ] = fd_ed25519_x8_base_enc;
  __mmask8 ok = fd_ed25519_x8_decode( &base, enc, (__mmask8)0xff );
  if( FD_UNLIKELY( ok!=(__mmask8)0xff ) ) FD_LOG_ERR(( "fd_ed25519_x8: base point failed to decode" ));

  fd_ed25519_x8_t p = base; /* [16^i] B, replicated across all lanes */

  for( int i=0; i<64; i++ ) {

    /* Lane m of "multiples" gets [m+1] p.  Every lane of "acc" holds the
       same value throughout, so lane m of the m-th partial sum is [m+1] p. */
    ulong mult[4][5][8];
    fd_ed25519_x8_t acc = p;
    fd_ed25519_x8_cached_t pc = fd_ed25519_x8_to_cached( &p );
    for( int m=0; m<8; m++ ) {
      if( m ) acc = fd_ed25519_x8_add( &acc, &pc );
      fd_ed25519_x8_scalar_pt_t s = fd_ed25519_x8_get_lane( &acc, 0 );
      for( int c=0; c<4; c++ ) for( int l=0; l<5; l++ ) mult[c][l][m] = s.c[c][l];
    }

    fd_ed25519_x8_t multiples;
    multiples.X = fd_r51x8_ld( (ulong const (*)[8])mult[0] );
    multiples.Y = fd_r51x8_ld( (ulong const (*)[8])mult[1] );
    multiples.Z = fd_r51x8_ld( (ulong const (*)[8])mult[2] );
    multiples.T = fd_r51x8_ld( (ulong const (*)[8])mult[3] );

    /* One batch inversion converts all eight multiples to affine. */
    fd_r51x8_t zinv = fd_r51x8_inv( multiples.Z );
    fd_r51x8_t ax   = fd_r51x8_reduce( fd_r51x8_mul( multiples.X, zinv ) );
    fd_r51x8_t ay   = fd_r51x8_reduce( fd_r51x8_mul( multiples.Y, zinv ) );

    fd_r51x8_t ypx = fd_r51x8_reduce( fd_r51x8_add( ay, ax ) );
    fd_r51x8_t ymx = fd_r51x8_reduce( fd_r51x8_sub( ay, ax ) );
    fd_r51x8_t t2d = fd_r51x8_reduce( fd_r51x8_mul( fd_r51x8_mul( ax, ay ),
                                                    fd_r51x8_bcast( fd_ed25519_x8_d2 ) ) );

    ulong sypx[5][8], symx[5][8], st2d[5][8];
    fd_r51x8_st( sypx, ypx );
    fd_r51x8_st( symx, ymx );
    fd_r51x8_st( st2d, t2d );
    for( int m=0; m<8; m++ ) {
      for( int l=0; l<5; l++ ) {
        fd_ed25519_x8_comb[i][m].ypx[l] = sypx[l][m];
        fd_ed25519_x8_comb[i][m].ymx[l] = symx[l][m];
        fd_ed25519_x8_comb[i][m].t2d[l] = st2d[l][m];
      }
    }

    /* p <- [16] p */
    for( int d=0; d<4; d++ ) p = fd_ed25519_x8_dbl( &p, d==3 );
  }
}

/* fd_ed25519_x8_selftest validates the hard coded constants against the
   curve itself, so a bad limb cannot silently produce a verifier that
   rejects everything (or, much worse, accepts something it should not). */

static void
fd_ed25519_x8_selftest( void ) {
  fd_r51x8_t sqrtm1 = fd_r51x8_bcast( fd_ed25519_x8_sqrtm1 );
  if( FD_UNLIKELY( fd_r51x8_eq( fd_r51x8_sqr( sqrtm1 ),
                                fd_r51x8_neg( fd_r51x8_one() ) )!=(__mmask8)0xff ) )
    FD_LOG_ERR(( "fd_ed25519_x8: sqrt(-1) constant is wrong" ));

  fd_r51x8_t d  = fd_r51x8_bcast( fd_ed25519_x8_d  );
  fd_r51x8_t d2 = fd_r51x8_bcast( fd_ed25519_x8_d2 );
  if( FD_UNLIKELY( fd_r51x8_eq( fd_r51x8_add( d, d ), d2 )!=(__mmask8)0xff ) )
    FD_LOG_ERR(( "fd_ed25519_x8: 2d constant is wrong" ));

  /* comb[0][0] is affine B; check -x^2 + y^2 == 1 + d x^2 y^2. */
  fd_r51x8_t ypx = fd_r51x8_bcast( fd_ed25519_x8_comb[0][0].ypx );
  fd_r51x8_t ymx = fd_r51x8_bcast( fd_ed25519_x8_comb[0][0].ymx );
  fd_r51x8_t two = fd_r51x8_add( fd_r51x8_one(), fd_r51x8_one() );
  fd_r51x8_t twoinv = fd_r51x8_inv( two );
  fd_r51x8_t bx  = fd_r51x8_mul( fd_r51x8_sub( ypx, ymx ), twoinv );
  fd_r51x8_t by  = fd_r51x8_mul( fd_r51x8_add( ypx, ymx ), twoinv );
  fd_r51x8_t x2  = fd_r51x8_sqr( bx );
  fd_r51x8_t y2  = fd_r51x8_sqr( by );
  fd_r51x8_t lhs = fd_r51x8_sub( y2, x2 );
  fd_r51x8_t rhs = fd_r51x8_add( fd_r51x8_one(),
                                 fd_r51x8_mul( d, fd_r51x8_mul( x2, y2 ) ) );
  if( FD_UNLIKELY( fd_r51x8_eq( lhs, rhs )!=(__mmask8)0xff ) )
    FD_LOG_ERR(( "fd_ed25519_x8: base point is not on the curve" ));
}

static void
fd_ed25519_x8_init( void ) {
  FD_ONCE_BEGIN {
    fd_ed25519_x8_build_comb();
    fd_ed25519_x8_selftest();
  } FD_ONCE_END;
}

/**********************************************************************/
/* Scalar recoding and table lookup                                   */
/**********************************************************************/

/* fd_ed25519_x8_radix16 expands a scalar below 2^253 into 64 signed
   radix-16 digits, each in [-8,8].  Signed digits halve the table size and
   let one table serve both signs via a cheap conditional negation. */

static inline void
fd_ed25519_x8_radix16( schar       out[ static 64 ],
                       uchar const s  [ static 32 ] ) {
  for( int i=0; i<32; i++ ) {
    out[ 2*i   ] = (schar)( s[i]     & 15 );
    out[ 2*i+1 ] = (schar)( (s[i]>>4)& 15 );
  }
  schar carry = 0;
  for( int i=0; i<63; i++ ) {
    schar v = (schar)( out[i] + carry );
    carry   = (schar)( (v + 8) >> 4 );
    out[i]  = (schar)( v - (schar)(carry << 4) );
  }
  out[63] = (schar)( out[63] + carry );
}

/* fd_ed25519_x8_select_cached picks, for every lane independently, one
   entry from that lane's own eight-entry multiples-of-A table.

   The table already lives in [entry][coordinate][limb][lane] order because
   it was built with eight-lane arithmetic, so a lane's entry is one masked
   512-bit load per (coordinate, limb).  Entry 0 is the identity, which is
   what makes a zero digit branch free.

   The digit is public (it comes from the challenge hash and the signature),
   so the variable indexing here leaks nothing. */

static inline fd_ed25519_x8_cached_t
fd_ed25519_x8_select_cached( fd_ed25519_x8_cached_t const table[ static 9 ],
                             schar const                  digit[ static 8 ] ) {
  fd_ed25519_x8_cached_t sel;
  __mmask8 neg = (__mmask8)0;

  __m512i acc[4][5];
  for( int c=0; c<4; c++ ) for( int i=0; i<5; i++ ) acc[c][i] = _mm512_setzero_si512();

  for( int lane=0; lane<8; lane++ ) {
    int   d = digit[ lane ];
    int   m = d<0 ? -d : d;
    if( d<0 ) neg = (__mmask8)( neg | (1<<lane) );
    __mmask8 k = (__mmask8)( 1<<lane );
    fd_ed25519_x8_cached_t const * e = &table[ m ];
    __m512i const * src[4] = { e->ypx.limb, e->ymx.limb, e->z.limb, e->t2d.limb };
    for( int c=0; c<4; c++ )
      for( int i=0; i<5; i++ )
        acc[c][i] = _mm512_mask_mov_epi64( acc[c][i], k, src[c][i] );
  }

  for( int i=0; i<5; i++ ) {
    sel.ypx.limb[i] = acc[0][i];
    sel.ymx.limb[i] = acc[1][i];
    sel.z  .limb[i] = acc[2][i];
    sel.t2d.limb[i] = acc[3][i];
  }

  /* Negating a cached point swaps Y+X with Y-X and negates 2dT. */
  fd_r51x8_t ypx = fd_r51x8_if( neg, sel.ymx, sel.ypx );
  fd_r51x8_t ymx = fd_r51x8_if( neg, sel.ypx, sel.ymx );
  sel.ypx = ypx;
  sel.ymx = ymx;
  sel.t2d = fd_r51x8_neg_if( neg, sel.t2d );
  return sel;
}

/* fd_ed25519_x8_select_comb gathers one shared base-point comb entry per
   lane.  Unlike the A table this one is scalar in memory (one copy, not
   one per lane), so the gather is a scalar transpose into the SoA layout.
   A zero digit yields the affine identity (1, 1, 0). */

static inline fd_ed25519_x8_precomp_t
fd_ed25519_x8_select_comb( int         position,
                           schar const digit[ static 8 ] ) {
  ulong ypx[5][8], ymx[5][8], t2d[5][8];
  __mmask8 neg = (__mmask8)0;

  for( int lane=0; lane<8; lane++ ) {
    int d = digit[ lane ];
    int m = d<0 ? -d : d;
    if( d<0 ) neg = (__mmask8)( neg | (1<<lane) );
    if( !m ) {
      for( int i=0; i<5; i++ ) { ypx[i][lane] = 0UL; ymx[i][lane] = 0UL; t2d[i][lane] = 0UL; }
      ypx[0][lane] = 1UL;
      ymx[0][lane] = 1UL;
      continue;
    }
    fd_ed25519_x8_comb_entry_t const * e = &fd_ed25519_x8_comb[ position ][ m-1 ];
    for( int i=0; i<5; i++ ) { ypx[i][lane] = e->ypx[i]; ymx[i][lane] = e->ymx[i]; t2d[i][lane] = e->t2d[i]; }
  }

  fd_ed25519_x8_precomp_t sel;
  for( int i=0; i<5; i++ ) {
    sel.ypx.limb[i] = _mm512_loadu_si512( (void const *)ypx[i] );
    sel.ymx.limb[i] = _mm512_loadu_si512( (void const *)ymx[i] );
    sel.t2d.limb[i] = _mm512_loadu_si512( (void const *)t2d[i] );
  }

  fd_r51x8_t p = fd_r51x8_if( neg, sel.ymx, sel.ypx );
  fd_r51x8_t m = fd_r51x8_if( neg, sel.ypx, sel.ymx );
  sel.ypx = p;
  sel.ymx = m;
  sel.t2d = fd_r51x8_neg_if( neg, sel.t2d );
  return sel;
}

/**********************************************************************/
/* The group equation                                                 */
/**********************************************************************/

/* fd_ed25519_x8_dsm computes [k]A + [s]B for eight independent (k, A, s)
   at once.

   [k]A uses a per-lane signed radix-16 window: 64 windows of four
   doublings and one addition, from an eight entry table of multiples of A
   that costs four doublings and three additions to build.

   [s]B uses the shared radix-16 comb, so it contributes 64 mixed additions
   and no doublings at all.  Both accumulators are then combined once. */

static fd_ed25519_x8_t
fd_ed25519_x8_dsm( fd_ed25519_x8_t const * a,
                   uchar const             k[ static 8*32 ],
                   uchar const             s[ static 8*32 ] ) {

  /* table[m] = [m] A, m in [0,8]; table[0] is the identity so that a zero
     digit needs no branch. */
  fd_ed25519_x8_cached_t table[9];
  {
    fd_ed25519_x8_t id = fd_ed25519_x8_identity();
    table[0] = fd_ed25519_x8_to_cached( &id );

    fd_ed25519_x8_t p1 = *a;
    fd_ed25519_x8_t p2 = fd_ed25519_x8_dbl( &p1, 1 );
    fd_ed25519_x8_t p4 = fd_ed25519_x8_dbl( &p2, 1 );
    fd_ed25519_x8_t p8 = fd_ed25519_x8_dbl( &p4, 1 );

    fd_ed25519_x8_cached_t c1 = fd_ed25519_x8_to_cached( &p1 );
    fd_ed25519_x8_cached_t c2 = fd_ed25519_x8_to_cached( &p2 );

    fd_ed25519_x8_t p3 = fd_ed25519_x8_add( &p2, &c1 );
    fd_ed25519_x8_t p6 = fd_ed25519_x8_dbl( &p3, 1 );
    fd_ed25519_x8_t p5 = fd_ed25519_x8_add( &p3, &c2 );
    fd_ed25519_x8_t p7 = fd_ed25519_x8_add( &p6, &c1 );

    table[1] = c1;
    table[2] = c2;
    table[3] = fd_ed25519_x8_to_cached( &p3 );
    table[4] = fd_ed25519_x8_to_cached( &p4 );
    table[5] = fd_ed25519_x8_to_cached( &p5 );
    table[6] = fd_ed25519_x8_to_cached( &p6 );
    table[7] = fd_ed25519_x8_to_cached( &p7 );
    table[8] = fd_ed25519_x8_to_cached( &p8 );
  }

  schar kd[64][8], sd[64][8];
  for( int lane=0; lane<8; lane++ ) {
    schar tmp[64];
    fd_ed25519_x8_radix16( tmp, k + 32*lane );
    for( int i=0; i<64; i++ ) kd[i][lane] = tmp[i];
    fd_ed25519_x8_radix16( tmp, s + 32*lane );
    for( int i=0; i<64; i++ ) sd[i][lane] = tmp[i];
  }

  /* [k]A, most significant window first. */
  fd_ed25519_x8_t q = fd_ed25519_x8_identity();
  for( int w=63; w>=0; w-- ) {
    if( FD_LIKELY( w!=63 ) ) {
      q = fd_ed25519_x8_dbl( &q, 0 );
      q = fd_ed25519_x8_dbl( &q, 0 );
      q = fd_ed25519_x8_dbl( &q, 0 );
      q = fd_ed25519_x8_dbl( &q, 1 );
    }
    fd_ed25519_x8_cached_t sel = fd_ed25519_x8_select_cached( table, kd[w] );
    q = fd_ed25519_x8_add( &q, &sel );
  }

  /* [s]B via the comb: no doublings, the position carries the weight. */
  fd_ed25519_x8_t r = fd_ed25519_x8_identity();
  for( int w=0; w<64; w++ ) {
    fd_ed25519_x8_precomp_t sel = fd_ed25519_x8_select_comb( w, sd[w] );
    r = fd_ed25519_x8_madd( &r, &sel );
  }

  fd_ed25519_x8_cached_t rc = fd_ed25519_x8_to_cached( &r );
  return fd_ed25519_x8_add( &q, &rc );
}

/**********************************************************************/
/* Encoding-level policy                                              */
/**********************************************************************/

/* fd_ed25519_x8_small_order classifies the 32-byte encodings that the
   permissive decoder maps onto one of the eight small-order points.  The
   decoder reduces y modulo p and ignores the sign bit for the purpose of
   the point's order, so this is an exact test, not a heuristic: the seven
   low-255-bit values are 0, 1, p-1, p, p+1 and the two order-8 y values. */

static inline int
fd_ed25519_x8_small_order( uchar const b[ static 32 ] ) {
  static uchar const alpha[32] = {
    0xc7,0x17,0x6a,0x70,0x3d,0x4d,0xd8,0x4f, 0xba,0x3c,0x0b,0x76,0x0d,0x10,0x67,0x0f,
    0x2a,0x20,0x53,0xfa,0x2c,0x39,0xcc,0xc6, 0x4e,0xc7,0xfd,0x77,0x92,0xac,0x03,0x7a
  };
  static uchar const neg_alpha[32] = {
    0x26,0xe8,0x95,0x8f,0xc2,0xb2,0x27,0xb0, 0x45,0xc3,0xf4,0x89,0xf2,0xef,0x98,0xf0,
    0xd5,0xdf,0xac,0x05,0xd3,0xc6,0x33,0x39, 0xb1,0x38,0x02,0x88,0x6d,0x53,0xfc,0x05
  };

  uchar diff;
  switch( b[0] ) {

  case 0x00: case 0x01:  /* y == 0 or y == 1 */
    diff = (uchar)( b[31] & 0x7f );
    for( int i=1; i<31; i++ ) diff = (uchar)( diff | b[i] );
    return !diff;

  case 0xec: case 0xed: case 0xee: /* y == p-1, p or p+1 */
    diff = (uchar)( (b[31] & 0x7f) ^ 0x7f );
    for( int i=1; i<31; i++ ) diff = (uchar)( diff | (uchar)(b[i] ^ 0xff) );
    return !diff;

  case 0x26: /* y == -alpha, an order-8 point */
    diff = (uchar)( (b[31] & 0x7f) ^ neg_alpha[31] );
    for( int i=0; i<31; i++ ) diff = (uchar)( diff | (uchar)(b[i] ^ neg_alpha[i]) );
    return !diff;

  case 0xc7: /* y == alpha, an order-8 point */
    diff = (uchar)( (b[31] & 0x7f) ^ alpha[31] );
    for( int i=0; i<31; i++ ) diff = (uchar)( diff | (uchar)(b[i] ^ alpha[i]) );
    return !diff;

  default:
    return 0;
  }
}

/**********************************************************************/
/* Batch verification                                                 */
/**********************************************************************/

#define FD_ED25519_X8_CHUNK (64UL) /* signatures per Montgomery inversion */

/* fd_ed25519_x8_finish converts the chunk's recomputed points to affine
   with ONE inversion in total and applies the comparison against R.

   The comparison is against R's y and sign bit rather than against the
   compressed bytes of the recomputed point, which reproduces
   fd_ed25519_verify's point equality exactly, including its acceptance of
   a non-canonical R encoding. */

static void
fd_ed25519_x8_finish( fd_ed25519_x8_t *     q,       /* [groups]  */
                      __mmask8 *            live,    /* [groups]  */
                      uchar const * const * sigs,
                      int *                 ok,
                      ulong                 base,
                      ulong                 cnt,
                      ulong                 groups ) {

  /* Montgomery: prefix[g] = Z[0]*..*Z[g-1], then one inversion unwinds
     every group's Z at three multiplies per group. */
  fd_r51x8_t prefix[ FD_ED25519_X8_CHUNK/8 ];
  fd_r51x8_t acc = fd_r51x8_one();
  for( ulong g=0; g<groups; g++ ) {
    prefix[g] = acc;
    acc = fd_r51x8_mul( acc, q[g].Z );
  }
  fd_r51x8_t inv = fd_r51x8_inv( acc );
  for( ulong gi=groups; gi>0; gi-- ) {
    ulong g = gi-1;
    fd_r51x8_t zinv = fd_r51x8_mul( inv, prefix[g] );
    inv = fd_r51x8_mul( inv, q[g].Z );

    fd_r51x8_t y = fd_r51x8_reduce( fd_r51x8_mul( q[g].Y, zinv ) );
    fd_r51x8_t x = fd_r51x8_mul( q[g].X, zinv );
    __mmask8   sgn = fd_r51x8_sgn( x );

    /* R's y, reduced the same permissive way the decoder would. */
    static uchar const zero_enc[32] = {0};
    ulong    rlimb[5][8];
    __mmask8 rsgn = (__mmask8)0;
    for( int lane=0; lane<8; lane++ ) {
      ulong index = base + g*8UL + (ulong)lane;
      uchar const * r = index<cnt ? sigs[ index ] : zero_enc;
      ulong l[5];
      fd_r51x8_unpack( l, r );
      for( int i=0; i<5; i++ ) rlimb[i][lane] = l[i];
      if( r[31] & 0x80 ) rsgn = (__mmask8)( rsgn | (1<<lane) );
    }
    fd_r51x8_t ry = fd_r51x8_reduce( fd_r51x8_ld( (ulong const (*)[8])rlimb ) );

    __mmask8 match = (__mmask8)( fd_r51x8_eq( y, ry ) & ~( sgn ^ rsgn ) & live[g] );

    for( int lane=0; lane<8; lane++ ) {
      ulong index = base + g*8UL + (ulong)lane;
      if( index>=cnt ) break;
      if( ok[ index ]!=FD_ED25519_SUCCESS ) continue; /* already rejected */
      ok[ index ] = ( (match>>lane)&1 ) ? FD_ED25519_SUCCESS : FD_ED25519_ERR_MSG;
    }
  }
}

int
fd_ed25519_verify_batch_x8( uchar const * const * msgs,
                            ulong         const * msg_sz,
                            uchar const * const * sigs,
                            uchar const * const * pubs,
                            int *                 ok,
                            ulong                 cnt ) {
  if( FD_UNLIKELY( !cnt ) ) return FD_ED25519_SUCCESS;
  fd_ed25519_x8_init();

  /* A group with a single live lane wastes seven lanes, and the fixed-width
     kernel then costs about what a full group does -- so one signature
     through this path is roughly twice fd_ed25519_verify's cost.  Peel a lone
     trailing signature off and verify it scalar.  The measured crossover is
     at two signatures, so only cnt%8==1 is worth splitting.

     This also makes the peeled signature's error code exactly
     fd_ed25519_verify's, including for an undecodable R. */
  ulong const orig_cnt = cnt;
  ulong batch_cnt = cnt;
  if( FD_UNLIKELY( (cnt % 8UL)==1UL ) ) {
    batch_cnt = cnt - 1UL;
    fd_sha512_t _tail_sha[1];
    fd_sha512_t * tail_sha = fd_sha512_join( fd_sha512_new( _tail_sha ) );
    ok[ batch_cnt ] = fd_ed25519_verify( msgs[ batch_cnt ], msg_sz[ batch_cnt ],
                                         sigs[ batch_cnt ], pubs[ batch_cnt ], tail_sha );
    fd_sha512_delete( fd_sha512_leave( tail_sha ) );
    if( FD_UNLIKELY( !batch_cnt ) )
      return ok[0]==FD_ED25519_SUCCESS ? FD_ED25519_SUCCESS : FD_ED25519_ERR_SIG;
  }
  cnt = batch_cnt;

  /* Staging for the multi-buffer hash: R || A || msg, one row per lane. */
  static FD_TL uchar stage[8][ 64UL + FD_ED25519_X8_FAST_MSG_MAX ] __attribute__((aligned(128)));
  fd_sha512_batch_t _batch[1] __attribute__((aligned(FD_SHA512_BATCH_ALIGN)));

  for( ulong base=0UL; base<cnt; base+=FD_ED25519_X8_CHUNK ) {
    ulong chunk  = fd_ulong_min( cnt-base, FD_ED25519_X8_CHUNK );
    ulong groups = (chunk+7UL)/8UL;

    fd_ed25519_x8_t q   [ FD_ED25519_X8_CHUNK/8 ];
    __mmask8        live[ FD_ED25519_X8_CHUNK/8 ];

    for( ulong g=0UL; g<groups; g++ ) {
      ulong lane0 = base + g*8UL;
      ulong n     = fd_ulong_min( cnt-lane0, 8UL );

      /* Encoding level policy first: it is cheap and it removes lanes
         before they cost any curve work. */
      __mmask8 active = (__mmask8)0;
      uchar const * aenc[8];
      for( ulong lane=0UL; lane<8UL; lane++ ) {
        if( lane>=n ) { aenc[lane] = NULL; continue; }
        ulong index = lane0 + lane;
        uchar const * sig = sigs[ index ];
        uchar const * pub = pubs[ index ];
        int err = FD_ED25519_SUCCESS;
        if     ( FD_UNLIKELY( !fd_curve25519_scalar_validate( sig+32 ) ) ) err = FD_ED25519_ERR_SIG;
        else if( FD_UNLIKELY( fd_ed25519_x8_small_order( pub ) )         ) err = FD_ED25519_ERR_PUBKEY;
        else if( FD_UNLIKELY( fd_ed25519_x8_small_order( sig ) )         ) err = FD_ED25519_ERR_SIG;
        ok[ index ] = err;
        if( FD_UNLIKELY( err!=FD_ED25519_SUCCESS ) ) { aenc[lane] = NULL; continue; }
        aenc[ lane ] = pub;
        active = (__mmask8)( active | (1<<lane) );
      }

      fd_ed25519_x8_t a;
      __mmask8 valid = fd_ed25519_x8_decode( &a, aenc, active );

      /* A lane whose public key does not decode is an invalid public key,
         exactly as in fd_ed25519_verify. */
      for( ulong lane=0UL; lane<n; lane++ )
        if( ( (active>>lane)&1 ) && !( (valid>>lane)&1 ) )
          ok[ lane0+lane ] = FD_ED25519_ERR_PUBKEY;

      /* Q = [k](-A) + [S]B.  Negating A rather than k matters: A can have
         order 8L, so reducing -k modulo L would not be equivalent. */
      a.X = fd_r51x8_neg( a.X );
      a.T = fd_r51x8_neg( a.T );

      uchar k[8][32] __attribute__((aligned(64)));
      uchar s[8][32] __attribute__((aligned(64)));
      memset( k, 0, sizeof(k) );
      memset( s, 0, sizeof(s) );

      uchar hash[8][64] __attribute__((aligned(128)));
      fd_sha512_batch_t * batch = fd_sha512_batch_init( _batch );
      int staged = 0;
      for( ulong lane=0UL; lane<8UL; lane++ ) {
        if( !( (valid>>lane)&1 ) ) continue;
        ulong index = lane0 + lane;
        ulong sz    = msg_sz[ index ];
        if( FD_LIKELY( sz<=FD_ED25519_X8_FAST_MSG_MAX ) ) {
          memcpy( stage[lane],      sigs[ index ],  32UL );
          memcpy( stage[lane]+32UL, pubs[ index ],  32UL );
          if( sz ) memcpy( stage[lane]+64UL, msgs[ index ], sz );
          batch = fd_sha512_batch_add( batch, stage[lane], 64UL+sz, hash[lane] );
          staged = 1;
        } else {
          /* Rare: hash long messages with the streaming implementation so
             the staging buffer stays a fixed, stack friendly size. */
          fd_sha512_t _sha[1];
          fd_sha512_t * sha = fd_sha512_join( fd_sha512_new( _sha ) );
          fd_sha512_fini( fd_sha512_append( fd_sha512_append( fd_sha512_append(
                          fd_sha512_init( sha ), sigs[index], 32UL ),
                          pubs[index], 32UL ), msgs[index], sz ), hash[lane] );
          fd_sha512_delete( fd_sha512_leave( sha ) );
        }
      }
      if( staged ) fd_sha512_batch_fini( batch ); else fd_sha512_batch_abort( batch );

      for( ulong lane=0UL; lane<8UL; lane++ ) {
        if( !( (valid>>lane)&1 ) ) continue;
        fd_curve25519_scalar_reduce( k[lane], hash[lane] );
        memcpy( s[lane], sigs[ lane0+lane ]+32, 32UL );
      }

      q   [g] = fd_ed25519_x8_dsm( &a, (uchar const *)k, (uchar const *)s );
      live[g] = valid;
    }

    fd_ed25519_x8_finish( q, live, sigs, ok, base, cnt, groups );
  }

  for( ulong i=0UL; i<batch_cnt; i++ )
    if( FD_UNLIKELY( ok[i]!=FD_ED25519_SUCCESS ) ) return FD_ED25519_ERR_SIG;
  if( FD_UNLIKELY( batch_cnt!=orig_cnt && ok[ batch_cnt ]!=FD_ED25519_SUCCESS ) )
    return FD_ED25519_ERR_SIG;
  return FD_ED25519_SUCCESS;
}

#endif /* FD_HAS_AVX512 */
