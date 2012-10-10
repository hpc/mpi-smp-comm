/**
 * Copyright (c) 2010-2012 Los Alamos National Security, LLC.
 *                         All rights reserved.
 *
 * This program was prepared by Los Alamos National Security, LLC at Los Alamos
 * National Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S.
 * Department of Energy (DOE). All rights in the program are reserved by the DOE
 * and Los Alamos National Security, LLC. Permission is granted to the public to
 * copy and use this software without charge, provided that this Notice and any
 * statement of authorship are reproduced on all copies. Neither the U.S.
 * Government nor LANS makes any warranty, express or implied, or assumes any
 * liability or responsibility for the use of this software.
 */

/**
 * @author Samuel K. Gutierrez
 */

/* /////////////////////////////////////////////////////////////////////////////
o CHANGE LOG

2010-11-05 Samuel K. Gutierrez
    * initial implementation.
2010-11-08 Samuel K. Gutierrez
    * updated algorithm - last version didn't work.
2011-04-22 Samuel K. Gutierrez
    * updated macros. 
2012-10-10 Samuel K. Gutierrez
    * updates for release.
///////////////////////////////////////////////////////////////////////////// */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <netdb.h>
#include <arpa/inet.h>
#include "mpi.h"

/* application name */
#define SMPCOMM_NAME "mpi-smp-comm"
/* current version */
#define SMPCOMM_VER  "0.0.5"

/* ////////////////////////////////////////////////////////////////////////// */
/* convenience macros                                                         */
/* ////////////////////////////////////////////////////////////////////////// */
/* "master" rank */
#define SMPCOMM_MASTER_RANK  0

#define SMPCOMM_FAILURE      0
#define SMPCOMM_SUCCESS      1

#define SMPCOMM_STRINGIFY(x) #x
#define SMPCOMM_TOSTRING(x)  SMPCOMM_STRINGIFY(x)

#define SMPCOMM_ERR_AT       __FILE__ " ("SMPCOMM_TOSTRING(__LINE__)")"
#define SMPCOMM_ERR_PREFIX   "-[SMPCOMM ERROR: "SMPCOMM_ERR_AT"]- "

/* error message */
#define SMPCOMM_ERR_MSG(pfargs...)                                             \
do {                                                                           \
    fprintf(stderr, SMPCOMM_ERR_PREFIX);                                       \
    fprintf(stderr, pfargs);                                                   \
} while (0)

/* mpi check */
#define SMPCOMM_MPICHK(_ret_,_gt_)                                             \
do {                                                                           \
    if (MPI_SUCCESS != (_ret_)) {                                              \
        MPI_Error_string((_ret_),                                              \
                         err_str,                                              \
                         &err_str_len);                                        \
        SMPCOMM_ERR_MSG("mpi success not returned on %s... %s (errno: %d)\n",  \
                         host_name_buff,                                       \
                         err_str,                                              \
                         (_ret_));                                             \
        goto _gt_;                                                             \
    }                                                                          \
} while (0)

/* printf with flush */
#define SMPCOMM_PF(pfargs...)                                                  \
do {                                                                           \
    fprintf(stdout, pfargs);                                                   \
    fflush(stdout);                                                            \
} while (0)

/* master rank printf */
/* master rank printf */
#define SMPCOMM_MPF(pfargs...)                                                 \
do {                                                                           \
    if (my_rank == (SMPCOMM_MASTER_RANK)) {                                    \
        fprintf(stdout, pfargs);                                               \
        fflush(stdout);                                                        \
    }                                                                          \
} while (0)

/* memory alloc check */
#define SMPCOMM_MEMCHK(_ptr_,_gt_)                                             \
do {                                                                           \
    if (NULL == (_ptr_)) {                                                     \
        SMPCOMM_ERR_MSG("memory allocation error on %s\n", host_name_buff);    \
        goto _gt_;                                                             \
    }                                                                          \
} while (0)

/* error string length */
static int err_str_len;
/* error string buffer */
static char err_str[MPI_MAX_ERROR_STRING];
/* host name buffer */
static char host_name_buff[MPI_MAX_PROCESSOR_NAME];

/* ////////////////////////////////////////////////////////////////////////// */
/* static forward declarations                                                */
/* ////////////////////////////////////////////////////////////////////////// */

/* ////////////////////////////////////////////////////////////////////////// */
/* helper functions                                                           */
/* ////////////////////////////////////////////////////////////////////////// */

/* ////////////////////////////////////////////////////////////////////////// */
static int
get_net_num(char *hstn,
            unsigned long int *net_num)
{
    struct hostent *host = NULL;

    if (NULL == (host = gethostbyname(hstn))) {
        SMPCOMM_ERR_MSG("gethostbyname error\n");
        /* epic fail! */
        goto err;
    }
    /* htonl used here because nodes could be different architectures */
    *net_num = htonl(inet_network(inet_ntoa(*(struct in_addr *)host->h_addr)));

    return SMPCOMM_SUCCESS;
err:
    return SMPCOMM_FAILURE;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
cmp_uli(const void *p1,
        const void *p2)
{
    return (*(unsigned long int *)p1 - *(unsigned long int *)p2);
}

/* ////////////////////////////////////////////////////////////////////////// */
static void
get_my_color(unsigned long int *net_nums         /* in  */,
             int net_num_len                     /* in  */,
             const unsigned long int *my_net_num /* in  */,
             int *my_color                       /* out */)
{
    int i      = 0;
    int node_i = 0;
    unsigned long int prev_num;
 
    qsort(net_nums, (size_t)net_num_len, sizeof(unsigned long int), cmp_uli);

    prev_num = net_nums[0];

    while (i < net_num_len && prev_num != *my_net_num) {
        while (net_nums[i] == prev_num) {
            ++i;
        }
        ++node_i;
        prev_num = net_nums[i];
    }

    *my_color = node_i;
}

/* ////////////////////////////////////////////////////////////////////////// */
/* main                                                                       */
/* ////////////////////////////////////////////////////////////////////////// */
int
main(int argc,
     char **argv)
{
    /* pointer to array that holds net nums for all rank processes */
    unsigned long int *net_nums = NULL;
    /* my color */
    int my_color;
    /* my rank */
    int my_rank;
    /* size of mpi_comm_world */
    int num_ranks;
    /* hostname length */
    int hostname_len;
    /* local rank communicator */
    MPI_Comm local_comm;
    /* holds mpi return codes */
    int mpi_ret_code;
    /* number of local ranks */
    int num_local_procs;
    /* my local rank number */
    int my_local_rank;
    unsigned long int net_num;

    /* init MPI */
    mpi_ret_code = MPI_Init(&argc, &argv);
    SMPCOMM_MPICHK(mpi_ret_code, error);
    /* get comm size */
    mpi_ret_code = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    SMPCOMM_MPICHK(mpi_ret_code, error);
    /* get my rank */
    mpi_ret_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    SMPCOMM_MPICHK(mpi_ret_code, error);
    /* get my host's name */
    mpi_ret_code = MPI_Get_processor_name(host_name_buff, &hostname_len);
    SMPCOMM_MPICHK(mpi_ret_code, error);

    /* get my network number */
    if (SMPCOMM_SUCCESS != get_net_num(host_name_buff, &net_num)) {
        goto error;
    }
    
    net_nums = (unsigned long int *)malloc(sizeof(unsigned long int) *
                                           num_ranks);
    SMPCOMM_MEMCHK(net_nums, error);

    /* get everyone else's net_num value */
    mpi_ret_code = MPI_Allgather(&net_num,
                                 1,
                                 MPI_UNSIGNED_LONG,
                                 net_nums,
                                 1,
                                 MPI_UNSIGNED_LONG,
                                 MPI_COMM_WORLD);
    SMPCOMM_MPICHK(mpi_ret_code, error);

    get_my_color(net_nums, num_ranks, &net_num, &my_color);

    /* free up some resources - no longer needed */
    free(net_nums);

    /* split into local node groups */
    mpi_ret_code = MPI_Comm_split(MPI_COMM_WORLD,
                                  my_color,
                                  my_rank,
                                  &local_comm);
    SMPCOMM_MPICHK(mpi_ret_code, error);

    /* get local comm size */
    mpi_ret_code = MPI_Comm_size(local_comm, &num_local_procs);
    SMPCOMM_MPICHK(mpi_ret_code, error);

    /* get my local comm rank */
    mpi_ret_code = MPI_Comm_rank(local_comm, &my_local_rank);
    SMPCOMM_MPICHK(mpi_ret_code, error);

    /* let the "master process" print out some header stuff */
    SMPCOMM_MPF("# %s %s\n", SMPCOMM_NAME, SMPCOMM_VER);
    SMPCOMM_MPF("# numpe %d\n", num_ranks);
   
    /**
     * not needed...  just used to make sure that the 
     * header stuff is flushed before following info
     */
    MPI_Barrier(MPI_COMM_WORLD);

    if (0 == my_local_rank) {
        SMPCOMM_PF("# host %s has %d local rank process%s\n",
                   host_name_buff,
                   num_local_procs,
                   num_local_procs> 1 && 0 !=  num_local_procs ? "es ":" ");
    }

    mpi_ret_code = MPI_Comm_free(&local_comm);
    SMPCOMM_MPICHK(mpi_ret_code, error);

    /* done! */
    mpi_ret_code = MPI_Finalize();
    SMPCOMM_MPICHK(mpi_ret_code, error);

    return EXIT_SUCCESS;
error:
    MPI_Abort(MPI_COMM_WORLD, mpi_ret_code);
    return EXIT_FAILURE;
}
