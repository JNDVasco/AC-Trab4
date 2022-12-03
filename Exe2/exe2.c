#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"

float conx = -0.74; // Real constant, horizontal axis (x)
float cony = 0.1;   // Imaginary constant, verital axis (y)
float Maxx = 2;     // Rightmost Real point of plane to be displayed
float Minx = -2;    // Leftmost Real point
float Maxy = 1;     // Uppermost Imaginary point
float Miny = -1;    // Lowermost Imaginary point
float initer = 300; // # of times to repeat function

float pixcorx; // 1 pixel on screen = this many units on the
float pixcory; // plane for both x and y axis'

int scrsizex; // Horizontal screen size in pixels
int scrsizey; // Vertical screen size

int resx;
int resy;
int *img;
int *imgAux;

int difIter;
float alfa;

char path[10] = "./images/";
char filename[64];
char buffer[20];

#define MASTER 0
int rank;
int size;
int start = 0;
int end = 0;

/**
 * Functions declarations
 */

struct timespec sum_timestamp(struct timespec begin, struct timespec end);
double time_between_timestamp(struct timespec begin, struct timespec end);
void julia(int xpt, int ypt);
void mandel(int xpt, int ypt);
void Generate(int frac, int starty, int endy);
void saveimg(int *img, int rx, int ry, char *fname);
void difusion(int starty, int endy);

void putpixel(int x, int y, int color)
{
    img[y * resx + x] = color * 1111;
}

int main(int argc, char **argv)
{
    struct timespec t1, t2;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc == 1)
    {
        resx = 640;
        resy = 480;
        difIter = 100;
        alfa = 0.5;
    }
    else if (argc == 5)
    {
        resx = atoi(argv[1]);
        resy = atoi(argv[2]);
        difIter = atoi(argv[3]);
        alfa = atof(argv[4]);
    }
    else
    {
        printf("Erro no número de argumentos\n");
        printf("Se não usar argumentos a imagem de saida terá dimensões 640x480\n");
        printf("Senão devera especificar o numero de colunas seguido do numero de linhas\n");
        printf("Seguido do numero de iteracoes da difusão e constante de difusao (entre 0 e 1)\n");
        printf("\nExemplo: %s 320 240 100 0.5\n", argv[0]);
        exit(1);
    }

    scrsizex = resx;
    scrsizey = resy;
    pixcorx = (Maxx - Minx) / scrsizex;
    pixcory = (Maxy - Miny) / scrsizey;

    // saveimg(img, resx, resy, "mandel.pgm");

    /**
     * MASTER CODE
     */
    if (rank == MASTER)
    {
        printf("I am a master %d of %d\n", rank, size);

        // Divide the Y res into smaller sizes for the slaves
        int step = ((scrsizey + size - 2) / (size - 1)); // Division round up
        printf("Step: %d\n", step);

        /**
         * Generate the mandel fractal
         *
         */

        // Send the data they need to compute to the slaves
        for (int i = 0; i < size - 1; i++)
        {
            start = i * step;
            end = ((start + step > scrsizey) ? scrsizey : start + step); // Check if the end is more than the max iterations

            MPI_Send(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);

            printf("Slave %d runs from %d to %d\n", i + 1, start, end);
        }

        // Allocate Space for the iamges
        img = (int *)malloc(resx * resy * sizeof(int));
        imgAux = (int *)malloc(resx * resy * sizeof(int));

        // Receive the images from the slaves and what they have done
        for (int i = 0; i < size - 1; i++)
        {
            MPI_Recv(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(imgAux, resx * resy, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            memcpy(img + start * resx, imgAux + start * resx, (end - start) * resx * sizeof(int));
        }

        saveimg(img, resx, resy, "mandel.ppm");

        /***
         * Mandel Diffusion
         */

        for (int x = 0; x < difIter; x++)
        {
            // Send the data they need to compute to the slaves
            for (int i = 0; i < size - 1; i++)
            {
                start = i * step;
                end = ((start + step > scrsizey) ? scrsizey : start + step); // Check if the end is more than the max iterations

                MPI_Send(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                MPI_Send(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                MPI_Send(img, resx * resy, MPI_INT, i + 1, 0, MPI_COMM_WORLD);

                // printf("Slave %d runs from %d to %d\n", i + 1, start, end);
            }

            // Receive the images from the slaves and what they have done

            for (int i = 0; i < size - 1; i++)
            {
                MPI_Recv(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(imgAux, resx * resy, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                memcpy(img + start * resx, imgAux + start * resx, (end - start) * resx * sizeof(int));
            }

            strcpy(filename, path);
            snprintf(buffer, 27, "Mandel_diff-%04d.ppm", x);
            strcat(filename, buffer);
            printf("[INFO]  %s\n", filename);
            saveimg(img, resx, resy, filename);
        }
        /**
         * Generate the julia fractal
         *
         */

        // Send the data they need to compute to the slaves
        for (int i = 0; i < size - 1; i++)
        {
            start = i * step;
            end = ((start + step > scrsizey) ? scrsizey : start + step); // Check if the end is more than the max iterations

            MPI_Send(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);

            printf("Slave %d runs from %d to %d\n", i + 1, start, end);
        }

        // Receive the images from the slaves and what they have done
        for (int i = 0; i < size - 1; i++)
        {
            MPI_Recv(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(imgAux, resx * resy, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            memcpy(img + start * resx, imgAux + start * resx, (end - start) * resx * sizeof(int));
        }

        saveimg(img, resx, resy, "julia.ppm");

        /***
         * Julia Diffusion
         */

        for (int x = 0; x < difIter; x++)
        {
            // Send the data they need to compute to the slaves
            for (int i = 0; i < size - 1; i++)
            {
                start = i * step;
                end = ((start + step > scrsizey) ? scrsizey : start + step); // Check if the end is more than the max iterations

                MPI_Send(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                MPI_Send(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                MPI_Send(img, resx * resy, MPI_INT, i + 1, 0, MPI_COMM_WORLD);

                // printf("Slave %d runs from %d to %d\n", i + 1, start, end);
            }

            // Receive the images from the slaves and what they have done

            for (int i = 0; i < size - 1; i++)
            {
                MPI_Recv(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(imgAux, resx * resy, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                memcpy(img + start * resx, imgAux + start * resx, (end - start) * resx * sizeof(int));
            }

            strcpy(filename, path);
            snprintf(buffer, 27, "Julia_diff-%04d.ppm", x);
            strcat(filename, buffer);
            printf("[INFO]  %s\n", filename);
            saveimg(img, resx, resy, filename);
        }
        free(img);
        free(imgAux);
    }
    /**
     * SLAVE CODE
     */
    else
    {
        img = (int *)malloc(resx * resy * sizeof(int));
        imgAux = (int *)malloc(resx * resy * sizeof(int));
        printf("I am a slave %d of %d\n", rank, size);

        /***
         * Mandel
         */

        MPI_Recv(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Generate(0, start, end);
        // Send image to master
        MPI_Send(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        MPI_Send(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        MPI_Send(img, resx * resy, MPI_INT, MASTER, 0, MPI_COMM_WORLD);

        /***
         * Mandel Difusion
         */
        // Receive the data they need to compute
        for (int i = 0; i < difIter; i++)
        {
            MPI_Recv(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(img, resx * resy, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            difusion(start, end);
            // Send image to master
            MPI_Send(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(imgAux, resx * resy, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }

        /***
         * Julia
         */

        MPI_Recv(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Generate(1, start, end);
        // Send image to master
        MPI_Send(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        MPI_Send(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        MPI_Send(img, resx * resy, MPI_INT, MASTER, 0, MPI_COMM_WORLD);

        /***
         * Julia Difusion
         */
        // Receive the data they need to compute
        for (int i = 0; i < difIter; i++)
        {
            MPI_Recv(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(img, resx * resy, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            difusion(start, end);
            // Send image to master
            MPI_Send(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(imgAux, resx * resy, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }

        free(img);
        free(imgAux);
    }

    MPI_Finalize();
    return 0;
}

/**
 * Julia
 */
void julia(int xpt, int ypt)
{
    long double x = xpt * pixcorx + Minx;
    long double y = Maxy - ypt * pixcory; // converting from pixels to points
    long double xnew = 0;
    long double ynew = 0;
    int k;

    for (k = 0; k <= initer; k++) // Each pixel loop
    {
        // The Julia Function Z=Z*Z+c (of complex numbers) into x and y parts
        xnew = x * x - y * y + conx;
        ynew = 2 * x * y + cony;
        x = xnew;
        y = ynew;
        if ((x * x + y * y) > 4)
            break; // Break condition Meaning the loop will go
                   // on to a value of infinity.
    }              // End each pixel loop

    int color = k;
    if (color > 15)
        color = color % 15;
    if (k >= initer)
        putpixel(xpt, ypt, 0);
    else
        putpixel(xpt, ypt, color);
}

/**
 * Mandelbrot
 */

void mandel(int xpt, int ypt)
{
    long double x = 0;
    long double y = 0; // converting from pixels to points
    long double xnew = 0;
    long double ynew = 0;
    int k;

    for (k = 0; k <= initer; k++) // Each pixel loop
    {
        // The Mandelbrot Function Z=Z*Z+c into x and y parts
        xnew = x * x - y * y + xpt * pixcorx + Minx;
        ynew = 2 * x * y + Maxy - ypt * pixcory;
        x = xnew;
        y = ynew;
        if ((x * x + y * y) > 4)
            break; // Break condition
    }              // End each pixel loop

    int color = k;
    if (color > 15)
        color = color % 15;
    if (k >= initer)
        putpixel(xpt, ypt, 0);
    else
        putpixel(xpt, ypt, color);
}

/**
 * Generate
 */

void Generate(int frac, int starty, int endy) // int startx, int endx,
{
    int thread_id, nloops, j = 0, i = 0;
#pragma omp parallel private(thread_id, nloops)
    {
        nloops = 0;
        thread_id = omp_get_thread_num();
#pragma omp for
        for (int j = starty; j < endy; j++)
        {
            if (nloops == 0)
                printf(" Thread %d started with j=%d\n", thread_id, j);
            ++nloops;

            for (int i = 0; i < scrsizex; i++)
            {
                // Start horizontal loop
                if (frac)
                {
                    julia(i, j);
                }
                else
                {
                    mandel(i, j);
                }
                // End horizontal loop
            }
            // End vertical loop
        }
    }
}

/**
 * Saveimg
 */

void saveimg(int *img, int rx, int ry, char *fname)
{
    FILE *fp;
    int color, i, j;
    fp = fopen(fname, "w");
    /* header for PPM output */
    fprintf(fp, "P6\n# CREATOR: AC Course, DEEC-UC\n");
    fprintf(fp, "%d %d\n255\n", rx, ry);

    for (i = 0; i < ry; i++)
    {
        for (j = 0; j < rx; j++)
        {

            color = img[i * rx + j];
            //	printf(" %d %d \n",img[i*rx+j],color);
            fputc((char)(color & 0x00ff), fp);
            fputc((char)(color >> 4 & 0x00ff), fp);
            fputc((char)((color >> 6 & 0x00ff)), fp);
        }
    }
    fclose(fp);
}

// Returns time in seconds
double time_between_timestamp(struct timespec begin, struct timespec end)
{
    struct timespec calc;
    calc.tv_sec = end.tv_sec - begin.tv_sec;
    calc.tv_nsec = end.tv_nsec - begin.tv_nsec;
    if (calc.tv_nsec < 0)
    {
        calc.tv_sec -= 1;
        calc.tv_nsec += 1e9;
    }
    return ((calc.tv_sec) + (calc.tv_nsec) / 1e9);
}

int returnPixVal(int i, int j)
{
    if (i < 0 || i >= resy || j < 0 || j >= resx)
    {
        return 0;
    }
    else
    {
        return img[i * resx + j];
    }
}

void difusion(int starty, int endy)
{
    int j = 0, i = 0, x = 0;
    int currentPixel = 0;
    int neighbourPixels = 0;

#pragma omp parallel private(j, i, currentPixel, neighbourPixels)
    {
#pragma omp for
        for (i = starty; i < endy; i++)
        {
            for (j = 0; j < resx; j++)
            {
                // imgAux[i * resx + j] = 0;
                neighbourPixels = 0;

                currentPixel = ((1 - alfa) * returnPixVal(i, j));
                neighbourPixels += returnPixVal(i - 1, j - 1);
                neighbourPixels += returnPixVal(i - 1, j);
                neighbourPixels += returnPixVal(i - 1, j + 1);
                neighbourPixels += returnPixVal(i, j - 1);
                neighbourPixels += returnPixVal(i, j + 1);
                neighbourPixels += returnPixVal(i + 1, j - 1);
                neighbourPixels += returnPixVal(i + 1, j);
                neighbourPixels += returnPixVal(i + 1, j + 1);
                neighbourPixels *= (alfa * 0.125);
                imgAux[i * resx + j] = currentPixel + neighbourPixels;
            }
        }
    }
}