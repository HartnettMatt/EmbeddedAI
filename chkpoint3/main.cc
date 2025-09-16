#include <unistd.h>
#include "main_functions.h"

int main(int argc, char* argv[]) {
  setup();
  while (true) {
    loop();
    usleep(1000000); // sleep for 1s
  }

}
