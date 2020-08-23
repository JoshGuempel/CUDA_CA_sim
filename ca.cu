#include <iostream>
#include <stdlib.h>
#include <thread>
#include <time.h>

#define BLOCK_SIZE 16

// wrap function for device code
__device__ int wrap_d(int N, int idx) {
  if (idx <= 0) {
    return N - 1;
  } else if (idx >= N - 1) {
    return 0;
  }
  return idx;
}

__global__ void updateUniverseKernel(const int width, const int height,
                                     const int *const oldUniverse,
                                     int *const newUniverse) {
  int i = (blockIdx.y * blockDim.y) + threadIdx.y;
  int j = (blockIdx.x * blockDim.x) + threadIdx.x;

  int numNeighbors =
      oldUniverse[wrap_d(height, i - 1) * width + wrap_d(width, j - 1)] +
      oldUniverse[wrap_d(height, i - 1) * width + wrap_d(width, j)] +
      oldUniverse[wrap_d(height, i - 1) * width + wrap_d(width, j + 1)] +
      oldUniverse[wrap_d(height, i) * width + wrap_d(width, j - 1)] +
      oldUniverse[wrap_d(height, i) * width + wrap_d(width, j + 1)] +
      oldUniverse[wrap_d(height, i + 1) * width + wrap_d(width, j - 1)] +
      oldUniverse[wrap_d(height, i + 1) * width + wrap_d(width, j)] +
      oldUniverse[wrap_d(height, i + 1) * width + wrap_d(width, j + 1)];

  if (oldUniverse[i * width + j] == 1 &&
      (numNeighbors <= 1 || numNeighbors >= 4)) { // on
    newUniverse[i * width + j] = 0;
  } else if (numNeighbors == 3 && oldUniverse[i * width + j] == 0) { // off
    newUniverse[i * width + j] = 1;
  } else {
    newUniverse[i * width + j] = oldUniverse[i * width + j];
  }
}

// N is assumed to be a multiple of BLOCK_SIZE
void updateUniverseGPU(const int width, const int height,
                       const int *const oldUniverse, int *const newUniverse) {
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);

  updateUniverseKernel<<<dimGrid, dimBlock>>>(width, height, oldUniverse,
                                              newUniverse);
  cudaDeviceSynchronize();
}

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
      std::cout << (universe[i * width + j] == 1 ? '@' : ' ');
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void profileRuntimeCPUvsGPU(const int width, const int height, int *&universe,
                            int *&universeTemp, const int numIterations) {
  clock_t t;

  t = clock();
  for (int i = 0; i < numIterations; i++) {
    updateUniverseCPU(width, height, universeTemp, universe);
    std::swap(universe, universeTemp);
  }
  t = clock() - t;

  float CPU_runtime_ms = (((float)t) / CLOCKS_PER_SEC) * 1000;

  t = clock();
  for (int i = 0; i < numIterations; i++) {
    updateUniverseGPU(width, height, universeTemp, universe);
    std::swap(universe, universeTemp);
  }
  t = clock() - t;

  float GPU_runtime_ms = (((float)t) / CLOCKS_PER_SEC) * 1000;

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "CPU Runtime: " << CPU_runtime_ms << " ms" << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "GPU Runtime: " << GPU_runtime_ms << " ms" << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
}

void runWithOutput(const int width, const int height, int *&universe,
                   int *&universeTemp, const int numIterations) {
  for (int i = 0; i < numIterations; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    system("clear");
    printUniverse(width, height, universeTemp);
    updateUniverseGPU(width, height, universeTemp, universe);
    std::swap(universe, universeTemp);
  }
}

int main() {
  int height = BLOCK_SIZE * 4;
  int width = BLOCK_SIZE * 16;

  int *universe;
  int *universeTemp;

  int numIterations = 0;
  std::string choice = "";

  srand(time(NULL));

  cudaMallocManaged(&universe, width * height * sizeof(int));
  cudaMallocManaged(&universeTemp, width * height * sizeof(int));

  populateUniverse(width, height, universeTemp);

  std::cout << "Enter P for profiling mode and O for output mode(O/P): ";
  std::cin >> choice;
  std::cout << "Enter number of iterations: ";
  std::cin >> numIterations;
  std::cout << "Starting..." << std::endl;

  if (choice == "P" || choice == "p") {
    profileRuntimeCPUvsGPU(width, height, universe, universeTemp,
                           numIterations);
  } else {
    runWithOutput(width, height, universe, universeTemp, numIterations);
  }

  cudaFree(universe);
  cudaFree(universeTemp);

  return 0;
}
