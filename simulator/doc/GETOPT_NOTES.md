#include <unistd.h>
int bflag, ch, fd;

bflag = 0;
while((ch = getopt(argc, argv, "bf:")) != -1) {
  switch (ch) {
  case 'b':
    bflag = 1;
    break;
  case 'f':
    if ((fd = open(optarg, O_RDONLY, 0)) < 0) {
      (void)fprintf(stderr, "myname: %s: %s\n", optarg, strerror(errno));
      exit(1);
    }
    break;
  case '?':
  default:
    usage();
  }
}
argc -= optind;
argv += optind;
