#include "../../../../util/fd_util.h"
/* TODO: Layering violation */
#include "../../../shared_dev/commands/bench/bench.h"

#include "../../fd_config.h"
#include "../../../platform/fd_cap_chk.h"
#include "../../../../disco/topo/fd_topo.h"
#include "../../../../disco/metrics/fd_metrics.h"

#include "helper.h"

#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <linux/capability.h>
#include <sys/ioctl.h>
#include <termios.h>
#include "generated/monitor_seccomp.h"

void
monitor_cmd_args( int *    pargc,
                  char *** pargv,
                  args_t * args ) {
  args->monitor.drain_output_fd = -1; /* only accessible to development commands, not the command line */
  args->monitor.dt_min          = fd_env_strip_cmdline_long( pargc, pargv, "--dt-min",   NULL,    6666667.          );
  args->monitor.dt_max          = fd_env_strip_cmdline_long( pargc, pargv, "--dt-max",   NULL,  133333333.          );
  args->monitor.duration        = fd_env_strip_cmdline_long( pargc, pargv, "--duration", NULL,          0.          );
  args->monitor.seed            = fd_env_strip_cmdline_uint( pargc, pargv, "--seed",     NULL, (uint)fd_tickcount() );
  args->monitor.ns_per_tic      = 1./fd_tempo_tick_per_ns( NULL ); /* calibrate during init */

  args->monitor.with_bench     = fd_env_strip_cmdline_contains( pargc, pargv, "--bench" );
  args->monitor.with_sankey    = fd_env_strip_cmdline_contains( pargc, pargv, "--sankey" );

  if( FD_UNLIKELY( args->monitor.dt_min<0L                   ) ) FD_LOG_ERR(( "--dt-min should be positive"          ));
  if( FD_UNLIKELY( args->monitor.dt_max<args->monitor.dt_min ) ) FD_LOG_ERR(( "--dt-max should be at least --dt-min" ));
  if( FD_UNLIKELY( args->monitor.duration<0L                 ) ) FD_LOG_ERR(( "--duration should be non-negative"    ));
}

void
monitor_cmd_perm( args_t *         args FD_PARAM_UNUSED,
                  fd_cap_chk_t *   chk,
                  config_t const * config ) {
  ulong mlock_limit = fd_topo_mlock( &config->topo );

  fd_cap_chk_raise_rlimit( chk, "monitor", RLIMIT_MEMLOCK, mlock_limit, "call `rlimit(2)` to increase `RLIMIT_MEMLOCK` so all memory can be locked with `mlock(2)`" );

  if( fd_sandbox_requires_cap_sys_admin( config->uid, config->gid ) )
    fd_cap_chk_cap( chk, "monitor", CAP_SYS_ADMIN,               "call `unshare(2)` with `CLONE_NEWUSER` to sandbox the process in a user namespace" );
  if( FD_LIKELY( getuid() != config->uid ) )
    fd_cap_chk_cap( chk, "monitor", CAP_SETUID,                  "call `setresuid(2)` to switch uid to the sanbox user" );
  if( FD_LIKELY( getgid() != config->gid ) )
    fd_cap_chk_cap( chk, "monitor", CAP_SETGID,                  "call `setresgid(2)` to switch gid to the sandbox user" );
}

typedef struct {
  ulong pid;
  ulong heartbeat;

  ulong in_backp;
  ulong backp_cnt;

  ulong nvcsw;
  ulong nivcsw;

  ulong regime_ticks[9];
} tile_snap_t;

typedef struct {
  ulong mcache_seq;

  ulong fseq_seq;

  ulong fseq_diag_tot_cnt;
  ulong fseq_diag_tot_sz;
  ulong fseq_diag_filt_cnt;
  ulong fseq_diag_filt_sz;
  ulong fseq_diag_ovrnp_cnt;
  ulong fseq_diag_ovrnr_cnt;
  ulong fseq_diag_slow_cnt;
} link_snap_t;

static ulong
tile_total_ticks( tile_snap_t * snap ) {
  ulong total = 0UL;
  for( ulong i=0UL; i<9UL; i++ ) total += snap->regime_ticks[ i ];
  return total;
}

static void
tile_snap( tile_snap_t *     snap_cur, /* Snapshot for each tile, indexed [0,tile_cnt) */
           fd_topo_t const * topo ) {
  for( ulong tile_idx=0UL; tile_idx<topo->tile_cnt; tile_idx++ ) {
    tile_snap_t * snap = &snap_cur[ tile_idx ];

    fd_topo_tile_t const * tile = &topo->tiles[ tile_idx ];
    snap->heartbeat = fd_metrics_tile( tile->metrics )[ FD_METRICS_GAUGE_TILE_HEARTBEAT_OFF ];

    fd_metrics_register( tile->metrics );

    FD_COMPILER_MFENCE();
    snap->pid       = FD_MGAUGE_GET( TILE, PID );
    snap->nvcsw     = FD_MCNT_GET( TILE, CONTEXT_SWITCH_VOLUNTARY_COUNT );
    snap->nivcsw    = FD_MCNT_GET( TILE, CONTEXT_SWITCH_INVOLUNTARY_COUNT );
    snap->in_backp  = FD_MGAUGE_GET( TILE, IN_BACKPRESSURE );
    snap->backp_cnt = FD_MCNT_GET( TILE, BACKPRESSURE_COUNT );
    for( ulong i=0UL; i<9UL; i++ ) {
      snap->regime_ticks[ i ] = fd_metrics_tl[ MIDX(COUNTER, TILE, REGIME_DURATION_NANOS)+i ];
    }
    FD_COMPILER_MFENCE();
  }
}

static ulong
find_producer_out_idx( fd_topo_t const *      topo,
                       fd_topo_tile_t const * producer,
                       fd_topo_tile_t const * consumer,
                       ulong                  consumer_in_idx ) {
  /* This finds all reliable consumers of the producers primary output,
     and then returns the position of the consumer (specified by tile
     and index of the in of that tile) in that list. The list ordering
     is not important, except that it matches the ordering of fseqs
     provided to fd_stem, so that metrics written for each link index
     are retrieved at the same index here.

     This is why we only count reliable links, because fd_stem only
     looks at and writes producer side diagnostics (is the link slow)
     for reliable links. */

  ulong reliable_cons_cnt = 0UL;
  for( ulong i=0UL; i<topo->tile_cnt; i++ ) {
    fd_topo_tile_t const * consumer_tile = &topo->tiles[ i ];
    for( ulong j=0UL; j<consumer_tile->in_cnt; j++ ) {
      for( ulong k=0UL; k<producer->out_cnt; k++ ) {
        if( FD_UNLIKELY( consumer_tile->in_link_id[ j ]==producer->out_link_id[ k ] && consumer_tile->in_link_reliable[ j ] ) ) {
          if( FD_UNLIKELY( consumer==consumer_tile && consumer_in_idx==j ) ) return reliable_cons_cnt;
          reliable_cons_cnt++;
        }
      }
    }
  }

  return ULONG_MAX;
}

static void
link_snap( link_snap_t *     snap_cur,
           fd_topo_t const * topo ) {
  ulong link_idx = 0UL;
  for( ulong tile_idx=0UL; tile_idx<topo->tile_cnt; tile_idx++ ) {
    for( ulong in_idx=0UL; in_idx<topo->tiles[ tile_idx ].in_cnt; in_idx++ ) {
      link_snap_t * snap = &snap_cur[ link_idx ];
      fd_frag_meta_t const * mcache = topo->links[ topo->tiles[ tile_idx ].in_link_id[ in_idx  ] ].mcache;
      ulong const * seq = (ulong const *)fd_mcache_seq_laddr_const( mcache );
      snap->mcache_seq = fd_mcache_seq_query( seq );

      ulong const * fseq = topo->tiles[ tile_idx ].in_link_fseq[ in_idx ];
      snap->fseq_seq = fd_fseq_query( fseq );

      ulong const * in_metrics = NULL;
      if( FD_LIKELY( topo->tiles[ tile_idx ].in_link_poll[ in_idx ] ) ) {
        in_metrics = (ulong const *)fd_metrics_link_in( topo->tiles[ tile_idx ].metrics, in_idx );
      }

      fd_topo_link_t const * link = &topo->links[ topo->tiles[ tile_idx ].in_link_id[ in_idx ] ];
      ulong producer_id = fd_topo_find_link_producer( topo, link );
      FD_TEST( producer_id!=ULONG_MAX );
      volatile ulong const * out_metrics = NULL;
      if( FD_LIKELY( topo->tiles[ tile_idx ].in_link_reliable[ in_idx ] ) ) {
        fd_topo_tile_t const * producer = &topo->tiles[ producer_id ];
        ulong cons_idx = find_producer_out_idx( topo, producer, &topo->tiles[ tile_idx ], in_idx );

        out_metrics = fd_metrics_link_out( producer->metrics, cons_idx );
      }
      FD_COMPILER_MFENCE();
      if( FD_LIKELY( in_metrics ) ) {
        snap->fseq_diag_tot_cnt   = in_metrics[ FD_METRICS_COUNTER_LINK_CONSUMED_COUNT_OFF ];
        snap->fseq_diag_tot_sz    = in_metrics[ FD_METRICS_COUNTER_LINK_CONSUMED_SIZE_BYTES_OFF ];
        snap->fseq_diag_filt_cnt  = in_metrics[ FD_METRICS_COUNTER_LINK_FILTERED_COUNT_OFF ];
        snap->fseq_diag_filt_sz   = in_metrics[ FD_METRICS_COUNTER_LINK_FILTERED_SIZE_BYTES_OFF ];
        snap->fseq_diag_ovrnp_cnt = in_metrics[ FD_METRICS_COUNTER_LINK_OVERRUN_POLLING_COUNT_OFF ];
        snap->fseq_diag_ovrnr_cnt = in_metrics[ FD_METRICS_COUNTER_LINK_OVERRUN_READING_COUNT_OFF ];
      } else {
        snap->fseq_diag_tot_cnt   = 0UL;
        snap->fseq_diag_tot_sz    = 0UL;
        snap->fseq_diag_filt_cnt  = 0UL;
        snap->fseq_diag_filt_sz   = 0UL;
        snap->fseq_diag_ovrnp_cnt = 0UL;
        snap->fseq_diag_ovrnr_cnt = 0UL;
      }

      if( FD_LIKELY( out_metrics ) )
        snap->fseq_diag_slow_cnt  = out_metrics[ FD_METRICS_COUNTER_LINK_SLOW_COUNT_OFF ];
      else
        snap->fseq_diag_slow_cnt  = 0UL;
      FD_COMPILER_MFENCE();
      snap->fseq_diag_tot_cnt += snap->fseq_diag_filt_cnt;
      snap->fseq_diag_tot_sz  += snap->fseq_diag_filt_sz;
      link_idx++;
    }
  }
}

/**********************************************************************/

static void write_stdout( char * buf, ulong buf_sz ) {
  ulong written = 0;
  ulong total = buf_sz;
  while( written < total ) {
    long n = write( STDOUT_FILENO, buf + written, total - written );
    if( FD_UNLIKELY( n < 0 ) ) {
      if( errno == EINTR ) continue;
      FD_LOG_ERR(( "error writing to stdout (%i-%s)", errno, fd_io_strerror( errno ) ));
    }
    written += (ulong)n;
  }
}

static int stop1 = 0;

#define FD_MONITOR_TEXT_BUF_SZ 131072
static char buffer[ FD_MONITOR_TEXT_BUF_SZ ];
static char buffer2[ FD_MONITOR_TEXT_BUF_SZ ];

static void
drain_to_buffer( char ** buf,
                 ulong * buf_sz,
                 int fd ) {
  while(1) {
    long nread = read( fd, buffer2, *buf_sz );
    if( FD_LIKELY( nread == -1 && errno == EAGAIN ) ) break; /* no data available */
    else if( FD_UNLIKELY( nread == -1 ) ) FD_LOG_ERR(( "read() failed (%i-%s)", errno, fd_io_strerror( errno ) ));

    char * ptr = buffer2;
    char * next;
    while(( next = memchr( ptr, '\n', (ulong)nread - (ulong)(ptr - buffer2) ))) {
      ulong len = (ulong)(next - ptr);
      if( FD_UNLIKELY( *buf_sz < len ) ) {
        write_stdout( buffer, FD_MONITOR_TEXT_BUF_SZ - *buf_sz );
        *buf = buffer;
        *buf_sz = FD_MONITOR_TEXT_BUF_SZ;
      }
      fd_memcpy( *buf, ptr, len );
      *buf += len;
      *buf_sz -= len;

      if( FD_UNLIKELY( *buf_sz < sizeof(TEXT_NEWLINE)-1 ) ) {
        write_stdout( buffer, FD_MONITOR_TEXT_BUF_SZ - *buf_sz );
        *buf = buffer;
        *buf_sz = FD_MONITOR_TEXT_BUF_SZ;
      }
      fd_memcpy( *buf, TEXT_NEWLINE, sizeof(TEXT_NEWLINE)-1 );
      *buf += sizeof(TEXT_NEWLINE)-1;
      *buf_sz -= sizeof(TEXT_NEWLINE)-1;

      ptr = next + 1;
    }
  }
}

static struct termios termios_backup;

static void
restore_terminal( void ) {
  (void)tcsetattr( STDIN_FILENO, TCSANOW, &termios_backup );
}

static void
run_monitor( config_t const * config,
             int              drain_output_fd,
             int              with_sankey,
             long             dt_min,
             long             dt_max,
             long             duration,
             uint             seed,
             double           ns_per_tic ) {
  fd_topo_t const * topo = &config->topo;

  /* Setup local objects used by this app */
  fd_rng_t _rng[1];
  fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, seed, 0UL ) );

  tile_snap_t * tile_snap_prv = (tile_snap_t *)fd_alloca( alignof(tile_snap_t), sizeof(tile_snap_t)*2UL*topo->tile_cnt );
  if( FD_UNLIKELY( !tile_snap_prv ) ) FD_LOG_ERR(( "fd_alloca failed" )); /* Paranoia */
  tile_snap_t * tile_snap_cur = tile_snap_prv + topo->tile_cnt;

  ulong link_cnt = 0UL;
  for( ulong tile_idx=0UL; tile_idx<topo->tile_cnt; tile_idx++ ) link_cnt += topo->tiles[ tile_idx ].in_cnt;
  link_snap_t * link_snap_prv = (link_snap_t *)fd_alloca( alignof(link_snap_t), sizeof(link_snap_t)*2UL*link_cnt );
  if( FD_UNLIKELY( !link_snap_prv ) ) FD_LOG_ERR(( "fd_alloca failed" )); /* Paranoia */
  link_snap_t * link_snap_cur = link_snap_prv + link_cnt;

  /* Get the initial reference diagnostic snapshot */
  tile_snap( tile_snap_prv, topo );
  link_snap( link_snap_prv, topo );
  long then; long tic; fd_tempo_observe_pair( &then, &tic );

  /* Monitor for duration ns.  Note that for duration==0, this
     will still do exactly one pretty print. */
  FD_LOG_NOTICE(( "monitoring --dt-min %li ns, --dt-max %li ns, --duration %li ns, --seed %u", dt_min, dt_max, duration, seed ));

  long stop = then + duration;
  if( duration == 0 ) stop = LONG_MAX;

#define PRINT( ... ) do {                                                       \
    int n = snprintf( buf, buf_sz, __VA_ARGS__ );                               \
    if( FD_UNLIKELY( n<0 ) ) FD_LOG_ERR(( "snprintf failed" ));                 \
    if( FD_UNLIKELY( (ulong)n>=buf_sz ) ) FD_LOG_ERR(( "snprintf truncated" )); \
    buf += n; buf_sz -= (ulong)n;                                               \
  } while(0)
  int monitor_pane = 0;

  /* Restore original terminal attributes at exit */
  atexit( restore_terminal );
  if( FD_UNLIKELY( 0!=tcgetattr( STDIN_FILENO, &termios_backup ) ) ) {
    FD_LOG_ERR(( "tcgetattr(STDIN_FILENO) failed (%i-%s)", errno, fd_io_strerror( errno ) ));
  }

  /* Disable character echo and line buffering */
  struct termios term = termios_backup;
  term.c_lflag &= (tcflag_t)~(ICANON | ECHO);
  if( FD_UNLIKELY( 0!=tcsetattr( STDIN_FILENO, TCSANOW, &term ) ) ) {
    FD_LOG_WARNING(( "tcsetattr(STDIN_FILENO) failed (%i-%s)", errno, fd_io_strerror( errno ) ));
  }

  for(;;) {
    /* Wait a somewhat randomized amount and then make a diagnostic
       snapshot */
    fd_log_wait_until( then + dt_min + (long)fd_rng_ulong_roll( rng, 1UL+(ulong)(dt_max-dt_min) ) );

    tile_snap( tile_snap_cur, topo );
    link_snap( link_snap_cur, topo );
    long now; long toc; fd_tempo_observe_pair( &now, &toc );

    /* Pretty print a comparison between this diagnostic snapshot and
       the previous one. */

    char * buf = buffer;
    ulong buf_sz = FD_MONITOR_TEXT_BUF_SZ;

    PRINT( "\033[2J\033[H" );

    /* drain any firedancer log messages into the terminal */
    if( FD_UNLIKELY( drain_output_fd >= 0 ) ) drain_to_buffer( &buf, &buf_sz, drain_output_fd );
    if( FD_UNLIKELY( buf_sz < FD_MONITOR_TEXT_BUF_SZ / 2 ) ) {
      /* make sure there's enough space to print the whole monitor in one go */
      write_stdout( buffer, FD_MONITOR_TEXT_BUF_SZ - buf_sz );
      buf = buffer;
      buf_sz = FD_MONITOR_TEXT_BUF_SZ;
    }

    if( FD_UNLIKELY( drain_output_fd >= 0 ) ) PRINT( TEXT_NEWLINE );
    int c = fd_getchar();
    if( FD_UNLIKELY( c=='\t'   ) ) monitor_pane = !monitor_pane;
    if( FD_UNLIKELY( c=='\x04' ) ) break; /* Ctrl-D */

    long dt = now-then;

    char now_cstr[ FD_LOG_WALLCLOCK_CSTR_BUF_SZ ];
    if( !monitor_pane ) {
      PRINT( "snapshot for %s | Use TAB to switch panes" TEXT_NEWLINE, fd_log_wallclock_cstr( now, now_cstr ) );
      PRINT( "    tile |     pid |      stale | heart | nivcsw              | nvcsw               | in backp |           backp cnt |  %% hkeep |  %% wait  |  %% backp | %% finish" TEXT_NEWLINE );
      PRINT( "---------+---------+------------+-------+---------------------+---------------------+----------+---------------------+----------+----------+----------+----------" TEXT_NEWLINE );
      for( ulong tile_idx=0UL; tile_idx<topo->tile_cnt; tile_idx++ ) {
        tile_snap_t * prv = &tile_snap_prv[ tile_idx ];
        tile_snap_t * cur = &tile_snap_cur[ tile_idx ];
        PRINT( " %7s", topo->tiles[ tile_idx ].name );
        PRINT( " | %7lu", cur->pid );
        PRINT( " | " ); printf_stale   ( &buf, &buf_sz, (long)(0.5+ns_per_tic*(double)(toc - (long)cur->heartbeat)), 1e8 /* 100 millis */ );
        PRINT( " | " ); printf_heart   ( &buf, &buf_sz, (long)cur->heartbeat, (long)prv->heartbeat  );
        PRINT( " | " ); printf_err_cnt ( &buf, &buf_sz, cur->nivcsw,          prv->nivcsw );
        PRINT( " | " ); printf_err_cnt ( &buf, &buf_sz, cur->nvcsw,           prv->nvcsw  );
        PRINT( " | " ); printf_err_bool( &buf, &buf_sz, cur->in_backp,        prv->in_backp   );
        PRINT( " | " ); printf_err_cnt ( &buf, &buf_sz, cur->backp_cnt,       prv->backp_cnt  );

        ulong cur_hkeep_ticks      = cur->regime_ticks[0]+cur->regime_ticks[1]+cur->regime_ticks[2];
        ulong prv_hkeep_ticks      = prv->regime_ticks[0]+prv->regime_ticks[1]+prv->regime_ticks[2];

        ulong cur_wait_ticks       = cur->regime_ticks[3]+cur->regime_ticks[6];
        ulong prv_wait_ticks       = prv->regime_ticks[3]+prv->regime_ticks[6];

        ulong cur_backp_ticks      = cur->regime_ticks[5];
        ulong prv_backp_ticks      = prv->regime_ticks[5];

        ulong cur_processing_ticks = cur->regime_ticks[4]+cur->regime_ticks[7];
        ulong prv_processing_ticks = prv->regime_ticks[4]+prv->regime_ticks[7];

        PRINT( " | " ); printf_pct( &buf, &buf_sz, cur_hkeep_ticks,      prv_hkeep_ticks,      0., tile_total_ticks( cur ), tile_total_ticks( prv ), DBL_MIN );
        PRINT( " | " ); printf_pct( &buf, &buf_sz, cur_wait_ticks,       prv_wait_ticks,       0., tile_total_ticks( cur ), tile_total_ticks( prv ), DBL_MIN );
        PRINT( " | " ); printf_pct( &buf, &buf_sz, cur_backp_ticks,      prv_backp_ticks,      0., tile_total_ticks( cur ), tile_total_ticks( prv ), DBL_MIN );
        PRINT( " | " ); printf_pct( &buf, &buf_sz, cur_processing_ticks, prv_processing_ticks, 0., tile_total_ticks( cur ), tile_total_ticks( prv ), DBL_MIN );
        PRINT( TEXT_NEWLINE );
      }
    } else {
      PRINT( "             link |  tot TPS |  tot bps | uniq TPS | uniq bps |   ha tr%% | uniq bw%% | filt tr%% | filt bw%% |           ovrnp cnt |           ovrnr cnt |            slow cnt |             tx seq" TEXT_NEWLINE );
      PRINT( "------------------+----------+----------+----------+----------+----------+----------+----------+----------+---------------------+---------------------+---------------------+-------------------" TEXT_NEWLINE );

      ulong link_idx = 0UL;
      for( ulong tile_idx=0UL; tile_idx<topo->tile_cnt; tile_idx++ ) {
        for( ulong in_idx=0UL; in_idx<topo->tiles[ tile_idx ].in_cnt; in_idx++ ) {
          link_snap_t * prv = &link_snap_prv[ link_idx ];
          link_snap_t * cur = &link_snap_cur[ link_idx ];

          fd_topo_link_t link = topo->links[ topo->tiles[ tile_idx ].in_link_id[ in_idx ] ];
          ulong producer_tile_id = fd_topo_find_link_producer( topo, &link );
          FD_TEST( producer_tile_id != ULONG_MAX );
          char const * producer = topo->tiles[ producer_tile_id ].name;
          PRINT( " %7s->%-7s", producer, topo->tiles[ tile_idx ].name );
          ulong cur_raw_cnt = /* cur->cnc_diag_ha_filt_cnt + */ cur->fseq_diag_tot_cnt;
          ulong cur_raw_sz  = /* cur->cnc_diag_ha_filt_sz  + */ cur->fseq_diag_tot_sz;
          ulong prv_raw_cnt = /* prv->cnc_diag_ha_filt_cnt + */ prv->fseq_diag_tot_cnt;
          ulong prv_raw_sz  = /* prv->cnc_diag_ha_filt_sz  + */ prv->fseq_diag_tot_sz;

          PRINT( " | " ); printf_rate( &buf, &buf_sz, 1e9, 0., cur_raw_cnt,             prv_raw_cnt,             dt );
          PRINT( " | " ); printf_rate( &buf, &buf_sz, 8e9, 0., cur_raw_sz,              prv_raw_sz,              dt ); /* Assumes sz incl framing */
          PRINT( " | " ); printf_rate( &buf, &buf_sz, 1e9, 0., cur->fseq_diag_tot_cnt,  prv->fseq_diag_tot_cnt,  dt );
          PRINT( " | " ); printf_rate( &buf, &buf_sz, 8e9, 0., cur->fseq_diag_tot_sz,   prv->fseq_diag_tot_sz,   dt ); /* Assumes sz incl framing */

          PRINT( " | " ); printf_pct ( &buf, &buf_sz, cur->fseq_diag_tot_cnt,  prv->fseq_diag_tot_cnt, 0.,
                                      cur_raw_cnt,             prv_raw_cnt,            DBL_MIN );
          PRINT( " | " ); printf_pct ( &buf, &buf_sz, cur->fseq_diag_tot_sz,   prv->fseq_diag_tot_sz,  0.,
                                      cur_raw_sz,              prv_raw_sz,             DBL_MIN ); /* Assumes sz incl framing */
          PRINT( " | " ); printf_pct ( &buf, &buf_sz, cur->fseq_diag_filt_cnt, prv->fseq_diag_filt_cnt, 0.,
                                      cur->fseq_diag_tot_cnt,  prv->fseq_diag_tot_cnt,  DBL_MIN );
          PRINT( " | " ); printf_pct ( &buf, &buf_sz, cur->fseq_diag_filt_sz,  prv->fseq_diag_filt_sz, 0.,
                                      cur->fseq_diag_tot_sz,   prv->fseq_diag_tot_sz,  DBL_MIN ); /* Assumes sz incl framing */

          PRINT( " | " ); printf_err_cnt( &buf, &buf_sz, cur->fseq_diag_ovrnp_cnt, prv->fseq_diag_ovrnp_cnt );
          PRINT( " | " ); printf_err_cnt( &buf, &buf_sz, cur->fseq_diag_ovrnr_cnt, prv->fseq_diag_ovrnr_cnt );
          PRINT( " | " ); printf_err_cnt( &buf, &buf_sz, cur->fseq_diag_slow_cnt,  prv->fseq_diag_slow_cnt  );
          PRINT( " | " ); printf_seq(     &buf, &buf_sz, cur->mcache_seq,          prv->mcache_seq  );
          PRINT( TEXT_NEWLINE );
          link_idx++;
        }
      }
    }
    if( FD_UNLIKELY( with_sankey ) ) {
      /* We only need to count from one of the benchs, since they both receive
         all of the transactions. */
      fd_topo_tile_t const * benchs = &topo->tiles[ fd_topo_find_tile( topo, "benchs", 0UL ) ];
      ulong fseq_sum = 0UL;
      for( ulong i=0UL; i<benchs->in_cnt; i++ ) {
        ulong const * fseq = benchs->in_link_fseq[ i ];
        fseq_sum += fd_fseq_query( fseq );
      }

      ulong net_tile_idx = fd_topo_find_tile( topo, "net", 0UL );
      if( FD_UNLIKELY( net_tile_idx==ULONG_MAX ) ) FD_LOG_ERR(( "net tile not found" ));

      fd_topo_tile_t const * net = &topo->tiles[ net_tile_idx ];
      ulong net_sent = fd_mcache_seq_query( fd_mcache_seq_laddr( topo->links[ net->out_link_id[ 0 ] ].mcache ) );
      net_sent      += fd_mcache_seq_query( fd_mcache_seq_laddr( topo->links[ net->out_link_id[ 1 ] ].mcache ) );
      net_sent = fseq_sum;

      ulong verify_failed  = 0UL;
      ulong verify_sent    = 0UL;
      ulong verify_overrun = 0UL;
      for( ulong i=0UL; i<config->layout.verify_tile_count; i++ ) {
        fd_topo_tile_t const * verify = &topo->tiles[ fd_topo_find_tile( topo, "verify", i ) ];
        verify_overrun += fd_metrics_link_in( verify->metrics, 0UL )[ FD_METRICS_COUNTER_LINK_OVERRUN_POLLING_FRAG_COUNT_OFF ] / config->layout.verify_tile_count;
        verify_failed += fd_metrics_link_in( verify->metrics, 0UL )[ FD_METRICS_COUNTER_LINK_FILTERED_COUNT_OFF ];
        verify_sent += fd_mcache_seq_query( fd_mcache_seq_laddr( topo->links[ verify->out_link_id[ 0 ] ].mcache ) );
      }

      fd_topo_tile_t const * dedup = &topo->tiles[ fd_topo_find_tile( topo, "dedup", 0UL ) ];
      ulong dedup_failed = 0UL;
      for( ulong i=0UL; i<config->layout.verify_tile_count; i++) {
        dedup_failed += fd_metrics_link_in( dedup->metrics, i )[ FD_METRICS_COUNTER_LINK_FILTERED_COUNT_OFF ];
      }
      ulong dedup_sent = fd_mcache_seq_query( fd_mcache_seq_laddr( topo->links[ dedup->out_link_id[ 0 ] ].mcache ) );

      fd_topo_tile_t const * pack = &topo->tiles[ fd_topo_find_tile( topo, "pack", 0UL ) ];
      volatile ulong * pack_metrics = fd_metrics_tile( pack->metrics );
      ulong pack_invalid = pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_WRITE_SYSVAR_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_ESTIMATION_FAIL_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_TOO_LARGE_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_EXPIRED_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_ADDR_LUT_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_UNAFFORDABLE_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_DUPLICATE_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_PRIORITY_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_NONVOTE_REPLACE_OFF ] +
                           pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_INSERTED_VOTE_REPLACE_OFF ];
      ulong pack_overrun = pack_metrics[ FD_METRICS_COUNTER_PACK_TRANSACTION_DROPPED_FROM_EXTRA_OFF ];
      ulong pack_sent = pack_metrics[ FD_METRICS_HISTOGRAM_PACK_TOTAL_TRANSACTIONS_PER_MICROBLOCK_COUNT_OFF + FD_HISTF_BUCKET_CNT ];

      static ulong last_fseq_sum;
      static ulong last_net_sent;
      static ulong last_verify_overrun;
      static ulong last_verify_failed;
      static ulong last_verify_sent;
      static ulong last_dedup_failed;
      static ulong last_dedup_sent;
      static ulong last_pack_overrun;
      static ulong last_pack_invalid;
      static ulong last_pack_sent;

      PRINT( "TXNS SENT:      %-10lu" TEXT_NEWLINE, fseq_sum );
      PRINT( "NET TXNS SENT:  %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, net_sent,       100.0 * (double)net_sent/(double)fseq_sum,        100.0 * (double)(net_sent - last_net_sent)/(double)(fseq_sum - last_fseq_sum)               );
      PRINT( "VERIFY OVERRUN: %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, verify_overrun, 100.0 * (double)verify_overrun/(double)net_sent,  100.0 * (double)(verify_overrun - last_verify_overrun)/(double)(net_sent - last_net_sent)   );
      PRINT( "VERIFY FAILED:  %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, verify_failed,  100.0 * (double)verify_failed/(double)net_sent,   100.0 * (double)(verify_failed - last_verify_failed)/(double)(net_sent - last_net_sent)     );
      PRINT( "VERIFY SENT:    %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, verify_sent,    100.0 * (double)verify_sent/(double)net_sent,     100.0 * (double)(verify_sent - last_verify_sent)/(double)(net_sent - last_net_sent)         );
      PRINT( "DEDUP FAILED:   %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, dedup_failed,   100.0 * (double)dedup_failed/(double)verify_sent, 100.0 * (double)(dedup_failed - last_dedup_failed)/(double)(verify_sent - last_verify_sent) );
      PRINT( "DEDUP SENT:     %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, dedup_sent,     100.0 * (double)dedup_sent/(double)verify_sent,   100.0 * (double)(dedup_sent - last_dedup_sent)/(double)(verify_sent - last_verify_sent)     );
      PRINT( "PACK OVERRUN:   %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, pack_overrun,   100.0 * (double)pack_overrun/(double)dedup_sent,  100.0 * (double)(pack_overrun - last_pack_overrun)/(double)(dedup_sent - last_dedup_sent)   );
      PRINT( "PACK INVALID:   %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, pack_invalid,   100.0 * (double)pack_invalid/(double)dedup_sent,  100.0 * (double)(pack_invalid - last_pack_invalid)/(double)(dedup_sent - last_dedup_sent)   );
      PRINT( "PACK SENT:      %-10lu %-5.2lf%%  %-5.2lf%%" TEXT_NEWLINE, pack_sent,      100.0 * (double)pack_sent/(double)dedup_sent,     100.0 * (double)(pack_sent - last_pack_sent)/(double)(dedup_sent - last_dedup_sent)         );

      last_fseq_sum = fseq_sum;
      last_net_sent = net_sent;
      last_verify_overrun = verify_overrun;
      last_verify_failed = verify_failed;
      last_verify_sent = verify_sent;
      last_dedup_failed = dedup_failed;
      last_dedup_sent = dedup_sent;
      last_pack_overrun = pack_overrun;
      last_pack_invalid = pack_invalid;
      last_pack_sent = pack_sent;
    }

    /* write entire monitor output buffer */
    write_stdout( buffer, sizeof(buffer) - buf_sz );

    if( FD_UNLIKELY( stop1 || (now-stop)>=0L ) ) {
      /* Stop once we've been monitoring for duration ns */
      break;
    }

    then = now; tic = toc;
    tile_snap_t * tmp = tile_snap_prv; tile_snap_prv = tile_snap_cur; tile_snap_cur = tmp;
    link_snap_t * tmp2 = link_snap_prv; link_snap_prv = link_snap_cur; link_snap_cur = tmp2;
  }
}

static void
signal1( int sig ) {
  (void)sig;
  exit( 0 ); /* gracefully exit */
}

void
monitor_cmd_fn( args_t *   args,
                config_t * config ) {
  if( FD_UNLIKELY( args->monitor.with_bench ) ) {
    add_bench_topo( &config->topo,
                    config->development.bench.affinity,
                    config->development.bench.benchg_tile_count,
                    config->development.bench.benchs_tile_count,
                    0UL,
                    0,
                    0.0f,
                    0.0f,
                    0UL,
                    0,
                    0U,
                    0,
                    0U,
                    1,
                    !config->is_firedancer );
  }

  struct sigaction sa = {
    .sa_handler = signal1,
    .sa_flags   = 0,
  };
  if( FD_UNLIKELY( sigaction( SIGTERM, &sa, NULL ) ) )
    FD_LOG_ERR(( "sigaction(SIGTERM) failed (%i-%s)", errno, fd_io_strerror( errno ) ));
  if( FD_UNLIKELY( sigaction( SIGINT, &sa, NULL ) ) )
    FD_LOG_ERR(( "sigaction(SIGINT) failed (%i-%s)", errno, fd_io_strerror( errno ) ));

  int allow_fds[ 5 ];
  ulong allow_fds_cnt = 0;
  allow_fds[ allow_fds_cnt++ ] = 0; /* stdin */
  allow_fds[ allow_fds_cnt++ ] = 1; /* stdout */
  allow_fds[ allow_fds_cnt++ ] = 2; /* stderr */
  if( FD_LIKELY( fd_log_private_logfile_fd()!=-1 && fd_log_private_logfile_fd()!=1 ) )
    allow_fds[ allow_fds_cnt++ ] = fd_log_private_logfile_fd(); /* logfile */
  if( FD_UNLIKELY( args->monitor.drain_output_fd!=-1 ) )
    allow_fds[ allow_fds_cnt++ ] = args->monitor.drain_output_fd; /* maybe we are interposing firedancer log output with the monitor */

  fd_topo_join_workspaces( &config->topo, FD_SHMEM_JOIN_MODE_READ_ONLY );

  struct sock_filter seccomp_filter[ 128UL ];
  uint drain_output_fd = args->monitor.drain_output_fd >= 0 ? (uint)args->monitor.drain_output_fd : (uint)-1;
  populate_sock_filter_policy_monitor( 128UL, seccomp_filter, (uint)fd_log_private_logfile_fd(), drain_output_fd );

  if( FD_UNLIKELY( close( config->log.lock_fd ) ) ) FD_LOG_ERR(( "close() failed (%i-%s)", errno, fd_io_strerror( errno ) ));

  if( FD_LIKELY( config->development.sandbox ) ) {
    fd_sandbox_enter( config->uid,
                      config->gid,
                      0,
                      0,
                      1, /* Keep controlling terminal for main so it can receive Ctrl+C */
                      0,
                      0UL,
                      0UL,
                      0UL,
                      allow_fds_cnt,
                      allow_fds,
                      sock_filter_policy_monitor_instr_cnt,
                      seccomp_filter );
  } else {
    fd_sandbox_switch_uid_gid( config->uid, config->gid );
  }

  fd_topo_fill( &config->topo );

  run_monitor( config,
               args->monitor.drain_output_fd,
               args->monitor.with_sankey,
               args->monitor.dt_min,
               args->monitor.dt_max,
               args->monitor.duration,
               args->monitor.seed,
               args->monitor.ns_per_tic );

  exit( 0 ); /* gracefully exit */
}

action_t fd_action_monitor = {
  .name           = "monitor",
  .args           = monitor_cmd_args,
  .fn             = monitor_cmd_fn,
  .require_config = 1,
  .perm           = monitor_cmd_perm,
  .description    = "Monitor a locally running Firedancer instance with a terminal GUI",
};
