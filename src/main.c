#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <gmp.h>
#include <ecm.h>

static void die(const char* m){ fprintf(stderr,"error: %s\n", m); exit(1); }

static int parse_mpz(mpz_t out, const char* s){
  return mpz_set_str(out, s, 0) == 0; /* base 0: dec or 0x... */
}

static void usage(const char* p){
  fprintf(stderr,
    "Usage:\n"
    "  %s ecm <N> [--B1 <float>] [--B2 <num>] [-c|--curves <n>] [--seed <u64>] [--threads <n>] [--S <int>] [--k <n>] [--ntt] [--sigma <u64>] [--verbose]\n"
    "  %s factor <N> [--B1 <float>] [-c|--curves <n>] [--maxsteps <n>]\n"
    "  %s --help\n", p,p,p);
}

/* ---------- quick trial division ---------- */
static int small_trial_div(mpz_t n, mpz_t f){
  if (mpz_divisible_ui_p(n, 2)) { mpz_set_ui(f,2); return 1; }
  for (unsigned long p=3; p<=100000UL; p+=2){
    if (mpz_divisible_ui_p(n, p)) { mpz_set_ui(f,p); return 1; }
  }
  return 0;
}

/* --- ECM tuning passed to ecm_once --- */
typedef struct {
  const char* B2_str;     /* nullable */
  long S;                 /* -1 => default */
  unsigned long k;        /* 0 => default */
  int use_ntt;            /* 0/1 */
  unsigned long sigma;    /* 0 => let libecm choose via RNG */
  int verbose;            /* 0/1 */
} ecm_tuning_t;

/* Stage-2 options */
static void set_stage2_params(ecm_params q, double B1, const ecm_tuning_t* t){
  if (!t) return;
  if (t->B2_str && t->B2_str[0]){
    mpz_t B2; mpz_init(B2);
    int ok = 0;
    if (strpbrk(t->B2_str, ".eE")){ double d = strtod(t->B2_str, NULL); if (d > 0){ mpz_set_d(B2, d); ok = 1; } }
    else ok = (mpz_set_str(B2, t->B2_str, 0) == 0);
    if (!ok){ mpz_clear(B2); die("invalid --B2"); }
    mpz_set(q->B2, B2);
    mpz_set_ui(q->B2min, (unsigned long)(B1 + 0.5));
    mpz_clear(B2);
  }
  if (t->S >= 0) q->S = (int)t->S;
  if (t->k) q->k = t->k;
  if (t->use_ntt) q->use_ntt = 1;
}

/* One ECM attempt (stage 1+2): ret>0 => factor in f (could be N), 0 => none */
static int ecm_once_raw(mpz_t n, double B1, mpz_t f, unsigned long seed, const ecm_tuning_t* t){
  ecm_params q; ecm_init(q);
  q->method = ECM_ECM;
  q->param  = ECM_PARAM_SUYAMA; /* default CLI parametrization */
  if (t && t->sigma) {
    /* NOTE: many small sigmas produce singular curves and libecm will complain. */
    mpz_set_ui(q->sigma, t->sigma);
  } else if (seed) {
    gmp_randseed_ui(q->rng, seed);   /* randomize curve choice */
  }
  set_stage2_params(q, B1, t);
  int ret = ecm_factor(f, n, B1, q);
  ecm_clear(q);
  return ret;
}

/* Make sure we don't return N; also, if sigma is fixed and unlucky, try a few nudges */
static int ecm_once_nontrivial(mpz_t n, double B1, mpz_t f, unsigned long seed, const ecm_tuning_t* t){
  /* If user fixed sigma, nudge it a few times to dodge invalid/singular */
  unsigned long base_sigma = (t && t->sigma) ? t->sigma : 0;
  for (int j=0; j < (base_sigma ? 8 : 1); ++j){
    ecm_tuning_t tt = t ? *t : (ecm_tuning_t){0};
    if (base_sigma) tt.sigma = base_sigma + (unsigned long)(j*193UL + 1UL);
    double b = B1;
    for (int shrink=0; shrink<=3; ++shrink){
      int r = ecm_once_raw(n, b, f, base_sigma ? 0 : seed, &tt);
      if (r > 0){
        if (mpz_cmp(f, n) == 0) { b *= 0.33; continue; } /* split “N found” case */
        if (mpz_cmp_ui(f,1) > 0 && mpz_cmp(f, n) < 0) return 1; /* good factor */
      }
      break; /* miss; change sigma or seed next */
    }
  }
  return 0;
}

static int ecm_loop(mpz_t n, double B1, unsigned long curves, mpz_t f, unsigned long seed, const ecm_tuning_t* t){
  for (unsigned long i=0; i<curves; ++i){
    unsigned long s = (t && t->sigma) ? 0 : (seed ? (seed + i) : (unsigned long)time(NULL) + i);
    if (ecm_once_nontrivial(n, B1, f, s, t)) return 1;
  }
  return 0;
}

/* ---------- Parallel ECM (for ecm subcommand) ---------- */
typedef struct {
  mpz_t n_local;
  double B1;
  unsigned long curves;
  unsigned long seed_base;
  mpz_t *shared_f;
  pthread_mutex_t *mtx;
  volatile int *stop;
  const ecm_tuning_t* tuning;
} worker_arg_t;

static void* worker_run(void *vp){
  worker_arg_t *a = (worker_arg_t*)vp;
  mpz_t f; mpz_init(f);
  for (unsigned long i=0; i<a->curves && !*(a->stop); ++i){
    unsigned long s = (a->tuning && a->tuning->sigma) ? 0 : (a->seed_base + i);
    if (ecm_once_nontrivial(a->n_local, a->B1, f, s, a->tuning)){
      pthread_mutex_lock(a->mtx);
      if (!*(a->stop)){ mpz_set(*(a->shared_f), f); *(a->stop) = 1; }
      pthread_mutex_unlock(a->mtx);
      break;
    }
  }
  mpz_clear(f);
  return NULL;
}

static int ecm_parallel(const mpz_t n, double B1, unsigned long curves,
                        unsigned threads, mpz_t f, unsigned long seed_base,
                        const ecm_tuning_t* t){
  if (threads <= 1) {
    mpz_t tmp; mpz_init_set(tmp, n);
    int r = ecm_loop(tmp, B1, curves, f, seed_base, t);
    mpz_clear(tmp);
    return r;
  }
  pthread_t *ths = (pthread_t*)calloc(threads, sizeof(pthread_t));
  worker_arg_t *args = (worker_arg_t*)calloc(threads, sizeof(worker_arg_t));
  if (!ths || !args) die("oom");

  unsigned long base = curves / threads, rem = curves % threads;
  pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  volatile int stop = 0;
  mpz_t shared_f; mpz_init(shared_f);

  for (unsigned tix=0; tix<threads; ++tix){
    unsigned long cnt = base + (tix < rem ? 1UL : 0UL);
    mpz_init_set(args[tix].n_local, n);
    args[tix].B1 = B1;
    args[tix].curves = cnt;
    args[tix].seed_base = (t && t->sigma) ? 0 : seed_base + (tix * 100000UL);
    args[tix].shared_f = &shared_f;
    args[tix].mtx = &mtx;
    args[tix].stop = &stop;
    args[tix].tuning = t;
    pthread_create(&ths[tix], NULL, worker_run, &args[tix]);
  }

  for (unsigned tix=0; tix<threads; ++tix){
    pthread_join(ths[tix], NULL);
    mpz_clear(args[tix].n_local);
  }

  int found = 0;
  if (stop){ mpz_set(f, shared_f); found = 1; }
  mpz_clear(shared_f);
  free(ths); free(args);
  return found;
}

/* ---------- Factor (recursive) ---------- */
static int is_probable_prime(const mpz_t n){ int r = mpz_probab_prime_p(n, 25); return r > 0; }

static void factor_rec(mpz_t n, double B1, unsigned long curves, unsigned maxsteps){
  if (mpz_cmp_ui(n,1)==0) return;
  if (is_probable_prime(n)){ gmp_printf("%Zd", n); return; }
  mpz_t f; mpz_init(f);
  if (small_trial_div(n, f)){
    mpz_t q; mpz_init(q); mpz_divexact(q, n, f);
    factor_rec(f, B1, curves, maxsteps); printf(" * "); factor_rec(q, B1, curves, maxsteps);
    mpz_clear(q); mpz_clear(f); return;
  }
  double b1 = B1;
  for (unsigned tries=0; tries<maxsteps; ++tries){
    if (ecm_loop(n, b1, curves, f, (unsigned long)time(NULL), NULL)){
      mpz_t q; mpz_init(q); mpz_divexact(q, n, f);
      factor_rec(f, B1, curves, maxsteps); printf(" * "); factor_rec(q, B1, curves, maxsteps);
      mpz_clear(q); mpz_clear(f); return;
    }
    b1 *= 3.0;
  }
  gmp_printf("[cofactor composite: %Zd]", n);
  mpz_clear(f);
}

int main(int argc, char** argv){
  if (argc < 2){ usage(argv[0]); return 1; }
  const char* cmd = argv[1];
  if (!strcmp(cmd, "--help") || !strcmp(cmd, "-h")){ usage(argv[0]); return 0; }

  if (!strcmp(cmd, "ecm")){
    if (argc < 3){ usage(argv[0]); return 1; }
    mpz_t n; mpz_init(n);
    if (!parse_mpz(n, argv[2])) die("invalid N");
    double B1 = 1e6; unsigned long curves = 50, seed = 0;
    unsigned threads = 1;
    ecm_tuning_t tuning = {0}; tuning.S = -1;

    for (int i=3;i<argc;i++){
      if (!strcmp(argv[i],"--B1") && i+1<argc) B1 = atof(argv[++i]);
      else if (!strcmp(argv[i],"--B2") && i+1<argc) tuning.B2_str = argv[++i];
      else if ((!strcmp(argv[i],"-c")||!strcmp(argv[i],"--curves")) && i+1<argc) curves = strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--seed") && i+1<argc) seed = strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--threads") && i+1<argc) threads = (unsigned)strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--S") && i+1<argc) tuning.S = strtol(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--k") && i+1<argc) tuning.k = strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--ntt")) tuning.use_ntt = 1;
      else if (!strcmp(argv[i],"--sigma") && i+1<argc) tuning.sigma = strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--verbose")) tuning.verbose = 1;
      else die("unknown arg");
    }

    /* Quick trial division (helps tiny N like 8051) */
    mpz_t f; mpz_init(f);
    if (small_trial_div(n, f)) { gmp_printf("%Zd\n", f); mpz_clear(f); mpz_clear(n); return 0; }

    if (tuning.verbose){
      fprintf(stderr, "ecm: B1=%.0f curves=%lu threads=%u seed=%lu",
              B1, curves, threads, seed);
      if (tuning.B2_str) fprintf(stderr, " B2=%s", tuning.B2_str);
      if (tuning.S>=0) fprintf(stderr, " S=%ld", tuning.S);
      if (tuning.k) fprintf(stderr, " k=%lu", tuning.k);
      if (tuning.use_ntt) fprintf(stderr, " ntt=1");
      if (tuning.sigma) fprintf(stderr, " sigma=%lu", tuning.sigma);
      fputc('\n', stderr);
    }

    int ok = ecm_parallel(n, B1, curves, threads, f, seed ? seed : (unsigned long)time(NULL), &tuning);
    if (ok) gmp_printf("%Zd\n", f); else printf("no-factor\n");
    mpz_clear(f); mpz_clear(n); return 0;
  }

  if (!strcmp(cmd, "factor")){
    if (argc < 3){ usage(argv[0]); return 1; }
    mpz_t n; mpz_init(n);
    if (!parse_mpz(n, argv[2])) die("invalid N");
    double B1=1e6; unsigned long curves=50; unsigned maxsteps=6;
    for (int i=3;i<argc;i++){
      if (!strcmp(argv[i],"--B1") && i+1<argc) B1=atof(argv[++i]);
      else if ((!strcmp(argv[i],"-c")||!strcmp(argv[i],"--curves")) && i+1<argc) curves=strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--maxsteps") && i+1<argc) maxsteps=(unsigned)strtoul(argv[++i],NULL,10);
      else die("unknown arg");
    }
    factor_rec(n, B1, curves, maxsteps); printf("\n"); mpz_clear(n); return 0;
  }

  usage(argv[0]); return 1;
}
