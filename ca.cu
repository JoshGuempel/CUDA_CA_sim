#include <iostream>
#include <stdlib.h>
#include <thread>
#include <time.h>

#define BLOCK_SIZE 16

// wrap function for cpu code
int wrap(int N, int idx) {
  if (idx <= 0) {
    return N - 1;
  } else if (idx >= N - 1) {
    return 0;
  }
  return idx;
}

void updateUniverseCPU(const int width, const int height,
                       const int *const oldUniverse, int *const newUniverse) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int numNeighbors =
          oldUniverse[wrap(height, i - 1) * width + wrap(width, j - 1)] +
          oldUniverse[wrap(height, i - 1) * width + wrap(width, j)] +
          oldUniverse[wrap(height, i - 1) * width + wrap(width, j + 1)] +
          oldUniverse[wrap(height, i) * width + wrap(width, j - 1)] +
          oldUniverse[wrap(height, i) * width + wrap(width, j + 1)] +
          oldUniverse[wrap(height, i + 1) * width + wrap(width, j - 1)] +
          oldUniverse[wrap(height, i + 1) * width + wrap(width, j)] +
          oldUniverse[wrap(height, i + 1) * width + wrap(width, j + 1)];

      if (oldUniverse[i * width + j] == 1 &&
          (numNeighbors <= 1 || numNeighbors >= 4)) { // on
        newUniverse[i * width + j] = 0;
      } else if (numNeighbors == 3 && oldUniverse[i * width + j] == 0) { // off
        newUniverse[i * width + j] = 1;
      } else {
        newUniverse[i * width + j] = oldUniverse[i * width + j];
      }
    }
  }
}

// Randomly add some cells that are living at the beginning
void populateUniverse(int width, int height, int *universe) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if ((rand() % 3) == 0) {
        universe[i * width + j] = 1;
      } else {
        universe[i * width + j] = 0;
      }
    }
  }
}

void printUniverse(const int width, const int height, int *universe) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << (universe[i * width + j] == 1 ? "#" : "-");
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  int height = BLOCK_SIZE * 4;
  int width = BLOCK_SIZE * 16;

  int *universe;
  int *universeTemp;

  srand(time(NULL));

  cudaMallocManaged(&universe, width * height * sizeof(int));
  cudaMallocManaged(&universeTemp, width * height * sizeof(int));

  populateUniverse(width, height, universeTemp);

  for (int i = 0; i < 400; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    system("clear");
    printUniverse(width, height, universeTemp);
    updateUniverseCPU(width, height, universeTemp, universe);
    int *temp = universeTemp;
    universeTemp = universe;
    universe = temp;
  }

  cudaFree(universe);
  cudaFree(universeTemp);

  std::cout << "done" << std::endl;
  return 0;
}
