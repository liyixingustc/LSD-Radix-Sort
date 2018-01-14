/**
 * @file    radix_sort.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 *
 * Copyright (c) 2016 Georgia Institute of Technology. All Rights Reserved.
 */

/*
 * TODO: implement your radix sort solution in this file
 */

#include <mpi.h>
#include <iostream>
#include <cmath>
#include "mystruct.h"

// returns the value of the digit starting at offset `offset` and containing `k` bits
#define GET_DIGIT(key, k, offset) ((key) >> (offset)) & ((1 << (k)) - 1)


/**
 * @brief   Parallel distributed radix sort.
 *
 * This function sorts the distributed input range [begin, end)
 * via lowest-significant-byte-first radix sort.
 *
 * This function will sort elements of type `T`, by the key of type `unsigned int`
 * which is returned by the key-access function `key_func`.
 *
 * The MPI datatype for the templated (general) type `T` has to be passed
 * via the `dt` parameter.
 *
 * @param begin         A pointer to the first element in the local range to be sorted.
 * @param end           A pointer to the end of the range to be sorted. This
 *                      pointer points one past the last element, such that the
 *                      total number of elements is given by `end - begin`.
 * @param key_func      A function with signature: `unsigned int (const T&)`.
 *                      This function returns the key of each element, which is
 *                      used for sorting.
 * @param dt            The MPI_Datatype which represents the type `T`. This
 *                      is used whenever elements of type `T` are communicated
 *                      via MPI.
 * @param comm          The communicator on which the sorting happens.
 *                      NOTE: this is not necessarily MPI_COMM_WORLD. Call
 *                            all MPI functions with this communicator and
 *                            NOT with MPI_COMM_WORLD.
 */
template <typename T>
void radix_sort(T* begin, T* end, unsigned int (*key_func)(const T&), MPI_Datatype dt, MPI_Comm comm, unsigned int k = 16) {
  // get comm rank and size

  int rank, p, i, j;
  int mark,mark_1;
  int error_code;
  int local_lrank;
  int send_mark;
  unsigned int current_key;
  int temp;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  int *local_hist;
  int *local_prefix;
  int *local_prefix_temp;
  int *global_hist;
  int *prefix_hist;
  int *send_counts;
  int *recv_counts;
  int *sdispls;
  int *rdispls;
  T *sendbuf;
  T *recvbuf;
  int *lrank; //local rank of elements before and after sending
  int *grank; //global rank of elements before sending
  int *dest; //destination of elements before sending
  int *lrank_temp; //helps get local rank of elements after sending

  // The number of elements per processor: n/p
  size_t np = end - begin;

  // the number of histogram buckets = 2^k
  unsigned int num_buckets = 1 << k;


  local_hist = new int[num_buckets];
  local_prefix= new int[np];
  local_prefix_temp= new int[num_buckets];
  global_hist = new int[num_buckets];
  prefix_hist = new int[num_buckets];
  send_counts = new int[p];
  recv_counts = new int[p];
  sdispls = new int[p];
  rdispls = new int[p];

  sendbuf = new T[np];
  recvbuf = new T[np];
  
  lrank = new int[np];
  lrank_temp = new int[np];
  grank = new int[np];
  dest = new int[np];


  for (unsigned int d = 0; d < 8*sizeof(unsigned int); d += k) {

    //for each run, initialize all arrays
    memset(local_hist, 0, num_buckets * sizeof(int));
    memset(local_prefix, 0, np * sizeof(int));
    memset(local_prefix_temp, 0, num_buckets * sizeof(int));
    memset(global_hist, 0, num_buckets * sizeof(int));
    memset(prefix_hist, 0, num_buckets * sizeof(int));
    memset(send_counts, 0, p * sizeof(int));
    memset(recv_counts, 0, p * sizeof(int));
    memset(sdispls, 0, p * sizeof(int));
    memset(rdispls, 0, p * sizeof(int));
    memset(lrank, 0, np * sizeof(int));
    memset(lrank_temp, 0, np * sizeof(int));
    memset(grank, 0, np * sizeof(int));
    memset(dest, 0, np * sizeof(int));



    // 1.) create histogram and sort via bucketing (~ counting sort)
    // create histogram
    memset(local_hist, 0, num_buckets * sizeof(int));
    memset(lrank_temp, 0, np * sizeof(int));
    for (i=0;i<np;i++) {
      //calculate the corresponding bucket mark should fit (mark)
      current_key=key_func(*(begin+i));
      current_key=GET_DIGIT(current_key, k, d);
      lrank_temp[i]=current_key;
      local_hist[current_key]++;
    }


    // calculate local rank by using histogram with local prefix sum
    memset(local_prefix, 0, np * sizeof(int));
    memset(local_prefix_temp, 0, num_buckets * sizeof(int));
    memset(lrank, 0, np * sizeof(int));
    for (i=0;i<np;i++){
      mark=lrank_temp[i];
      local_prefix_temp[mark]++;
      local_prefix[i]=local_prefix_temp[mark];

      for (j=0;j<mark;j++){
	lrank[i]=lrank[i]+local_hist[j];
      }
      lrank[i]=lrank[i]+local_prefix_temp[mark];
    }
 
    // 2.) get global histograms (P, G) via MPI_Exscan/MPI_Allreduce,...
    //Calculate total sum
    error_code=MPI_Barrier(comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_alltoallv\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }

    error_code=MPI_Allreduce(local_hist,global_hist,num_buckets,MPI_INT,MPI_SUM,comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_Allreduce\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }
    MPI_Barrier(comm);


    //Calculate prefix sum
    memset(prefix_hist, 0, num_buckets * sizeof(int));

    error_code=MPI_Barrier(comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_alltoallv\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }

    error_code=MPI_Exscan(local_hist,prefix_hist,num_buckets,MPI_INT,MPI_SUM,comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_Exscan\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }


    //Get rank for each element
    memset(send_counts, 0, p * sizeof(int));
    for (i=0;i<np;i++) {

      local_lrank=lrank_temp[i];
      mark_1=0;
      for (j=0;j<local_lrank;j++) {
	mark_1=mark_1+global_hist[j];
      }
      mark_1=mark_1+prefix_hist[local_lrank];
      mark_1=mark_1+local_prefix[i];

      grank[i]=mark_1-1;
      dest[i]=ceil(grank[i]/np);
      send_counts[dest[i]]++;
    }

    // 4.) communicate send_counts to get recv_counts
    // alltoall
    error_code=MPI_Barrier(comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_alltoallv\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }

    error_code=MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_alltoall\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }

    // 5.) calculate displacements for send and recv
    sdispls[0]=0;
    for (i=1;i<p;i++){
      sdispls[i]=sdispls[i-1]+send_counts[i-1];
    }

    rdispls[0]=0;
    for (i=1;i<p;i++){
      rdispls[i]=rdispls[i-1]+recv_counts[i-1];
    }

    //arrange sendbuf
    send_mark=0;
    for (i=0;i<np;i++){
      *(sendbuf+lrank[i]-1)=begin[i];
    }


    // 6.) MPI_Alltoallv
    error_code=MPI_Barrier(comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_alltoallv\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }
    error_code=MPI_Alltoallv(sendbuf, send_counts, sdispls, dt, recvbuf, recv_counts, rdispls, dt, comm);
    if (error_code != MPI_SUCCESS) {
      fprintf(stderr, "%4d, error code: 4%d in MPI_alltoallv\n", rank, error_code);
      MPI_Abort(comm,error_code);
    }
    //    std::cout<<"Alltoallv success barrier"<<std::endl;

    // 7.)local sorting via bucketing (~ counting sort)    
    // create new histogram
    memset(local_hist, 0, num_buckets * sizeof(int));
    memset(lrank_temp, 0, np * sizeof(int));
    memset(local_prefix, 0, np * sizeof(int));
    memset(local_prefix_temp, 0, num_buckets * sizeof(int));
    memset(lrank, 0, np * sizeof(int));
    for (i=0;i<np;i++) {
      //calculate the corresponding bucket mark should fit (mark)
      current_key=key_func(*(recvbuf+i));
      mark=0;
      current_key=GET_DIGIT(current_key, k, d);
      lrank_temp[i]=current_key;
      local_hist[current_key]++;
    }

    // calculate local rank by using histogram with local prefix sum   
    for (i=0;i<np;i++){
      mark=lrank_temp[i];
      local_prefix_temp[mark]++;
      local_prefix[i]=local_prefix_temp[mark];

      for (j=0;j<mark;j++){
	lrank[i]=lrank[i]+local_hist[j];
      }
      lrank[i]=lrank[i]+local_prefix_temp[mark];
    }
 
    for (i=0;i<np;i++){
      j=lrank[i]-1;
      *(begin+j)=recvbuf[i];
    }
    MPI_Barrier(comm);
  }

  //release allocated arrays
  delete []local_hist;
  delete []local_prefix;
  delete []local_prefix_temp;
  delete []global_hist;
  delete []prefix_hist;
  delete []send_counts;
  delete []recv_counts;
  delete []sdispls;
  delete []rdispls;
  delete []sendbuf;
  delete []recvbuf;
  delete []lrank;
  delete []grank;
  delete []dest;
  delete []lrank_temp;

  //  std::cout<<"finish"<<std::endl;
  for(i=0;i<np;i++){
    temp=key_func(*(begin+i));
    //    std::cout<<rank<<" "<<i<<" "<<temp<<std::endl;
  }

}

