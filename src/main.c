#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gmp.h>
#include <ecm.h>

static void die(const char* m){ fprintf(stderr,"error: %s\n", m); exit(1); }

static int parse_mpz(mpz_t out, const char* s){
  /* base=0 lets 0x... (hex) or decimal both work */
  return mpz_set_str(out, s, 0) == 0;
}

static void usage(const char* p){
  fprintf(stderr,
    "Usage:\n"
    "  %s ecm <N> [--B1 <float>] [-c|--curves <n>] [--seed <u64>]\n"
    "  %s factor <N> [--B1 <float>] [-c|--curves <n>] [--maxsteps <n>]\n"
    "  %s --help\n", p,p,p);
}

static int small_trial_div(mpz_t n, mpz_t f){
  if (mpz_divisible_ui_p(n, 2)) { mpz_set_ui(f,2); return 1; }
  for (unsigned long p=3; p<=100000UL; p+=2){
    if (mpz_divisible_ui_p(n, p)) { mpz_set_ui(f,p); return 1; }
  }
  return 0;
}

static int ecm_once(mpz_t n, double B1, mpz_t f, unsigned long seed){
  ecm_params q; ecm_init(q);
  if (seed) gmp_randseed_ui(q->rng, seed);
  int ret = ecm_factor(f, n, B1, q);
  ecm_clear(q);
  return ret; /* >0 => factor found; 0 => none; <0 => error */
}

static int ecm_loop(mpz_t n, double B1, unsigned long curves, mpz_t f, unsigned long seed){
  for (unsigned long i=0; i<curves; ++i){
    int r = ecm_once(n, B1, f, seed ? (seed+i) : 0);
    if (r > 0) return 1;
  }
  return 0;
}

static int is_probable_prime(const mpz_t n){
  int r = mpz_probab_prime_p(n, 25);
  return r > 0; /* 1 or 2 => probably/definitely prime */
}

static void factor_rec(mpz_t n, double B1, unsigned long curves, unsigned maxsteps){
  if (mpz_cmp_ui(n,1)==0) return;
  if (is_probable_prime(n)){ gmp_printf("%Zd", n); return; }

  mpz_t f; mpz_init(f);
  if (small_trial_div(n, f)){
    mpz_t q; mpz_init(q); mpz_divexact(q, n, f);
    factor_rec(f, B1, curves, maxsteps); printf(" * ");
    factor_rec(q, B1, curves, maxsteps);
    mpz_clear(q); mpz_clear(f); return;
  }

  /* Try ECM with progressive B1 if needed */
  double b1 = B1;
  for (unsigned tries=0; tries<maxsteps; ++tries){
    if (ecm_loop(n, b1, curves, f, (unsigned long)time(NULL))){
      mpz_t q; mpz_init(q); mpz_divexact(q, n, f);
      factor_rec(f, B1, curves, maxsteps); printf(" * ");
      factor_rec(q, B1, curves, maxsteps);
      mpz_clear(q); mpz_clear(f); return;
    }
    b1 *= 3.0; /* ramp B1 */
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
    for (int i=3;i<argc;i++){
      if (!strcmp(argv[i],"--B1") && i+1<argc) B1 = atof(argv[++i]);
      else if ((!strcmp(argv[i],"-c")||!strcmp(argv[i],"--curves")) && i+1<argc) curves = strtoul(argv[++i],NULL,10);
      else if (!strcmp(argv[i],"--seed") && i+1<argc) seed = strtoul(argv[++i],NULL,10);
      else die("unknown arg");
    }
    mpz_t f; mpz_init(f);
    if (ecm_loop(n,B1,curves,f,seed)) gmp_printf("%Zd\n", f); else printf("no-factor\n");
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
