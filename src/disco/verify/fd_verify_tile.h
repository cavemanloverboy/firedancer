#ifndef HEADER_fd_src_disco_verify_fd_verify_tile_h
#define HEADER_fd_src_disco_verify_fd_verify_tile_h

/* The verify tile verifies that the cryptographic signatures of
   incoming transactions match the data being signed.  Transactions with
   invalid signatures are filtered out of the frag stream. */

#include "../topo/fd_topo.h"
#include "../metrics/generated/fd_metrics_enums.h"

/* FD_BATCH_VERIFY enables cross-transaction signature batching via
   fd_ed25519_verify_batch_x8.  Requires AVX-512.  When disabled (the
   default), the tile verifies one transaction at a time with
   fd_ed25519_verify_batch_single_msg. */

#ifndef FD_BATCH_VERIFY
#define FD_BATCH_VERIFY 0
#endif

#if FD_BATCH_VERIFY
# if !FD_HAS_AVX512
#   error "FD_BATCH_VERIFY requires FD_HAS_AVX512"
# endif
# define FD_VERIFY_BATCH_MAX (8UL)
#endif

#define FD_TXN_VERIFY_SUCCESS  0
#define FD_TXN_VERIFY_FAILED  -1
#define FD_TXN_VERIFY_DEDUP   -2

extern fd_topo_run_tile_t fd_tile_verify;

/* fd_verify_in_ctx_t is a context object for each in (producer) mcache
   connected to the verify tile. */

typedef struct {
  fd_wksp_t * mem;
  ulong       chunk0;
  ulong       wmark;
} fd_verify_in_ctx_t;

#if FD_BATCH_VERIFY
/* Staged transaction waiting for a batched verify.  Payload lives in
   the out dcache at out_chunk; pointers below alias into that slot. */
typedef struct {
  ulong         out_chunk;
  ulong         realized_sz;
  ulong         tsorig;
  ulong         ha_dedup_tag;
  int           is_bundle;
  int           dedup;
  uchar         sig_cnt;
  uchar const * msg;
  ulong         msg_sz;
  uchar const * signatures; /* sig_cnt * 64 */
  uchar const * pubkeys;    /* sig_cnt * 32 */
} fd_verify_batch_slot_t;
#endif

typedef struct {
  /* TODO switch to fd_sha512_batch_t? */
  fd_sha512_t * sha[ FD_TXN_ACTUAL_SIG_MAX ];

  int   bundle_failed;
  ulong bundle_id;

  ulong round_robin_idx;
  ulong round_robin_cnt;

  ulong   tcache_depth;
  ulong   tcache_map_cnt;
  ulong * tcache_sync;
  ulong * tcache_ring;
  ulong * tcache_map;

  ulong              in_kind[ 32 ];
  fd_verify_in_ctx_t in[ 32 ];

  fd_wksp_t * out_mem;
  ulong       out_chunk0;
  ulong       out_wmark;
  ulong       out_chunk;

  ulong       hashmap_seed;

#if FD_BATCH_VERIFY
  fd_verify_batch_slot_t batch[ FD_VERIFY_BATCH_MAX ];
  ulong                  batch_cnt;
  /* Per-in producer activity.  before_frag sets saw_frag[in]=1 (keep or
     RR skip); AFTER_POLL_IDLE sets saw_frag[in]=0 when that in is
     caught up.  after_credit flushes only when !saw_frag[batch_in_idx],
     so empty polls on other ins do not end a quic batch. */
  int                    saw_frag[ 32 ];
  ulong                  batch_in_idx;
#endif

  struct {
    ulong verify_tile_result[ FD_METRICS_ENUM_VERIFY_TILE_RESULT_CNT ];
    ulong gossiped_votes_cnt;
    /* FD_BATCH_VERIFY flush-size histograms (index = size-1).  Zero
       when batching is disabled.  batch_sig_cnt[7] counts flushes with
       8 or more signature lanes. */
    ulong batch_txn_cnt[ FD_METRICS_ENUM_VERIFY_BATCH_SIZE_CNT ];
    ulong batch_sig_cnt[ FD_METRICS_ENUM_VERIFY_BATCH_SIZE_CNT ];
  } metrics;
} fd_verify_ctx_t;

static inline int
fd_txn_verify( fd_verify_ctx_t * ctx,
               uchar const *     udp_payload,
               ushort const      payload_sz,
               fd_txn_t const *  txn,
               int               dedup,
               ulong *           opt_sig ) {

  /* We do not want to deref any non-data field from the txn struct more than once */
  uchar  signature_cnt = txn->signature_cnt;
  ushort signature_off = txn->signature_off;
  ushort acct_addr_off = txn->acct_addr_off;
  ushort message_off   = txn->message_off;

  uchar const * signatures = udp_payload + signature_off;
  uchar const * pubkeys = udp_payload + acct_addr_off;
  uchar const * msg = udp_payload + message_off;
  ulong msg_sz = (ulong)payload_sz - message_off;

  /* The first signature is the transaction id, i.e. a unique identifier.
     So use this to do a quick dedup of ha traffic. */

  ulong ha_dedup_tag = fd_hash( ctx->hashmap_seed, signatures, 64UL );
  int ha_dup = 0;
  if( FD_LIKELY( dedup ) ) {
    FD_FN_UNUSED ulong tcache_map_idx = 0; /* ignored */
    FD_TCACHE_QUERY( ha_dup, tcache_map_idx, ctx->tcache_map, ctx->tcache_map_cnt, ha_dedup_tag );
    if( FD_UNLIKELY( ha_dup ) ) {
      return FD_TXN_VERIFY_DEDUP;
    }
  }

  /* Verify signatures */
  int res = fd_ed25519_verify_batch_single_msg( msg, msg_sz, signatures, pubkeys, ctx->sha, signature_cnt );
  if( FD_UNLIKELY( res != FD_ED25519_SUCCESS ) ) {
    return FD_TXN_VERIFY_FAILED;
  }

  /* Insert into the tcache to dedup ha traffic.
     The dedup check is repeated to guard against duped txs verifying signatures at the same time */
  if( FD_LIKELY( dedup ) ) {
    FD_TCACHE_INSERT( ha_dup, *ctx->tcache_sync, ctx->tcache_ring, ctx->tcache_depth, ctx->tcache_map, ctx->tcache_map_cnt, ha_dedup_tag );
    if( FD_UNLIKELY( ha_dup ) ) {
      return FD_TXN_VERIFY_DEDUP;
    }
  }

  *opt_sig = ha_dedup_tag;
  return FD_TXN_VERIFY_SUCCESS;
}

#endif /* HEADER_fd_src_disco_verify_fd_verify_tile_h */
