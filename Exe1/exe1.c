/**
 * Include
 */

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

//  These are the global variables.  That means they can be changed from
//  any of my functions.  This is usfull for my options function.  That
//  means that these variables are 'user' variables and can be changed
//  and the changes will be stored for the displaying part of the program

float conx = -0.74; // Real constant, horizontal axis (x)
float cony = 0.1;   // Imaginary constant, verital axis (y)
float Maxx = 2;     // Rightmost Real point of plane to be displayed
float Minx = -2;    // Leftmost Real point
float Maxy = 1;     // Uppermost Imaginary point
float Miny = -1;    // Lowermost Imaginary point

float pixcorx; // 1 pixel on screen = this many units on the
float pixcory; // plane for both x and y axis'

int initer = 150; // # of times to repeat function

int scrsizex; // Horizontal screen size in pixels
int scrsizey; // Vertical screen size

int resx;
int resy;
int *img;

#define MASTER 0

/*
 img = [0,1,...,resx*resy-1]

 img = [0,1,2,3,4,5,6,7,8]

 i ->
 j	0 1 2
||	3 4 5
\/	6 7 8

    i + j * 3

    1 + 1 * 3 = 4

 Pixel(i,j) = img[i + j*resx]


*/

char path[10] = "./images/";
char filename[64];
char buffer[20];

void putpixel(int x, int y, int color);
void julia(int xpt, int ypt, int maxIter);
void mandel(int xpt, int ypt, int maxIter);
void Generate(int frac, int start, int end);
void saveimg(int *img, int rx, int ry, char *fname);

int main(int argc, char **argv)
{
    int rank;
    int size;
    int start = 0;
    int end = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc == 1)
    {
        resx = 640;
        resy = 480;
    }

    else if (argc == 3)
    {
        resx = atoi(argv[1]);
        resy = atoi(argv[2]);
    }
    else
    {
        printf("Erro no número de argumentos\n");
        printf("Se não usar argumentos a imagem de saida terá dimensões 640x480\n");
        printf("Senão devera especificar o numero de colunas seguido do numero de linhas\n");
        printf("\nExemplo: %s 320 240\n", argv[0]);
        exit(1);
    }

    scrsizex = resx;
    scrsizey = resy;
    pixcorx = (Maxx - Minx) / scrsizex;
    pixcory = (Maxy - Miny) / scrsizey;

    if(rank == MASTER)
    {
        printf("I am a master %d of %d\n", rank, size);

        int step = ((initer + size - 2) / (size - 1)); // Division round up
        printf("Step: %d\n", step);

        for (int i = 0; i < size - 1; i++)
        {
            start = i * step;
            end = ((start + step > initer) ? initer : start + step); //Check if the end is more than the max iterations

            MPI_Send(&start, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);

            printf("Slave %d runs from %d to %d\n", i + 1, start, end);
        }
    }
    else
    {
        MPI_Recv(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("I am a slave %d of %d and I run from %d to %d\n", rank, size, start, end);
        printf("I am a slave %d of %d and I run from %d to %d\n", rank, size, initer - end, initer - start);

        img = (int *)malloc(resx * resy * sizeof(int));
        Generate(0, start, end);
        Generate(1, initer - end, initer - start);
        free(img);
    }



    MPI_Finalize();
    return 0;
}

/**
 * Generate function
 * If frac = 0 -> mandel
 * If frac = 1 -> Julia
 *
 */
void Generate(int frac, int start, int end)
{
    for (int maxIter = start; maxIter < end; maxIter++) // Each pixel loop
    {
        int j = 0;
        do // Start vertical loop
        {
            int i = 0;
            do // Start horizontal loop
            {
                if (frac)
                {
                    julia(i, j, maxIter);
                }
                else
                {
                    mandel(i, j, maxIter);
                }
                i++;
            } while ((i < scrsizex)); // End horizontal loop
            j++;
        } while ((j < scrsizey)); // End vertical loop

        strcpy(filename, path);

        if (frac)
        {
            snprintf(buffer, 20, "julia-%05d.ppm", maxIter);
        }
        else
        {
            snprintf(buffer, 20, "mandel-%05d.ppm", maxIter);
        }

        strcat(filename, buffer);
        saveimg(img, resx, resy, filename);
        printf("%s\n", filename);
    }
}

/**
 * Fractal generating functions
 * Recieves the x y point to compute the fractal and the max iteration
 */
// Julia
void julia(int xpt, int ypt, int maxIter)
{
    long double x = xpt * pixcorx + Minx;
    long double y = Maxy - ypt * pixcory; // converting from pixels to points
    long double xnew = 0;
    long double ynew = 0;
    int k;

    for (k = 0; k <= maxIter; k++) // Each pixel loop
    {
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
// Mandel
void mandel(int xpt, int ypt, int maxIter)
{
    long double x = 0;
    long double y = 0; // converting from pixels to points
    long double xnew = 0;
    long double ynew = 0;
    int k = 0;

    for (k = 0; k <= maxIter; k++) // Each pixel loop
    {
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

// Simple function to put a pixel col
void putpixel(int x, int y, int color)
{
    img[y * resx + x] = color * 1111;
    // printf("x: %d y: %d color: %d img: %d\n", x, y, color, img[y * resx + x]);
}

/**
 * Save the image
 * Recieves the img array, the x y size and the filename
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
            // printf(" %d %d \n",img[i*rx+j],color);
            fputc((char)(color & 0x00ff), fp);
            fputc((char)(color >> 4 & 0x00ff), fp);
            fputc((char)((color >> 6 & 0x00ff)), fp);
        }
    }
    fclose(fp);
}
