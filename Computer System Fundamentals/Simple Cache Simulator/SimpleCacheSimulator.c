/**
 * @file csim.c
 * @brief this program aims to implement a simple cache simulator with LRU
 * replacement policy and follow the write back, write allocate policy.
 * This simulator is able to simulate the behavior of a cache memory with
 * arbitrary size and asociativity.
 * It takes in the following command-line arguments: -h, -v, -s, -E, -b, -t.
 * Usage could be found by inputting -h and -v will print out detailed
 * results of the simulation.
 * It output the total number of hits, misses, evictions, dirty bytes and dirty
 * bytes that haven been evicted at the end.
 * @author: Yiru Xiong
 **/

#include "cachehelper.h"
#include <getopt.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Information about a cache */
typedef struct {
    unsigned long long valid; // valid bit in a cache line
    unsigned long long tag;   // tag bit
    unsigned long long block; // block bit
    int dirty;                // dirty bit for implementing writing back
    int timer;                // time stamp for implementing LRU replacement
} line, *set, **cache;

int sum_dirty_bytes; // store the total of dirty bytes that have been written
int h = 0, v = 0, s = 0, E = 0, b = 0, S = 0,
    B = 0;       // store inputs from command line
char *tracefile; // tracefile info
/* initialize fields for updating counts and storing results */
csim_stats_t res; // predefined struct to store simulation statistics
const csim_stats_t *stats = &res; // argument of printSummary function(const)

/* helper functions defined */
void read_traces(cache newcache);
void cache_update(cache newcache, char operator, unsigned long long address);
void LRU_timer(set newset);
void help_mode();

/* This function aims to read trace files and parse operations */
void read_traces(cache newcache) {
    // initialize fields needed
    char operator;
    unsigned long long address;
    int size;

    // open and read a trace file
    FILE *traces = fopen(tracefile, "r");
    // check if file has been successfully open
    if ((h != 1) && (traces == NULL)) {
        printf("Unable to open trace files!\n");
        exit(1);
    }

    // retrieve operator, address and size information from trace file
    while ((fscanf(traces, "%c %llx,%d", &operator, &address, &size)) > 0) {
        // if operator is I, do nothing
        switch (operator) {
        case 'I':
            continue; // no I in testing trace files. Added for completeness
        case 'L':
            // add verbosity mode(for -v function)
            if (v == 1) {
                printf("L, %llx,%d", address, size);
            }
            cache_update(newcache, operator, address);
            if (v == 1) {
                printf("\n");
            }
            break;
        case 'S':
            if (v == 1) {
                printf("S, %llx,%d", address, size);
            }
            cache_update(newcache, operator, address);
            if (v == 1) {
                printf("\n");
            }
            break;
        case 'M':
            cache_update(newcache, operator, address);
            cache_update(newcache, operator, address);
            break; // no M in testing trace files. Added for completeness
        default:
            break;
        }
    }
    // close the current file
    fclose(traces);
}

/* update time stamp for lines to implement LRU replacement policy*/
void LRU_timer(set newset) {
    for (int ll = 0; ll < E; ll += 1) {
        if (newset[ll].valid == 1) {
            newset[ll].timer += 1;
        }
    }
}

/* This function aims to update cache statistics(hits, misses, evictions, dirty
 * bytes, dirty bytes evicted) after each operations in tracefiles */
void cache_update(cache newcache, char operator, unsigned long long address) {
    unsigned long long addrinfo, tagbit, setbit;
    int LRU_base;
    int LRU_idx = 0;
    long mask = (~0) << s; // mask for setbit
    set newset;

    // get address
    addrinfo = address >> b;
    // get tag
    tagbit = addrinfo >> s;
    // get set bits
    setbit = addrinfo & (~mask);
    newset = newcache[setbit]; // go to the corresponding set
    LRU_base = newset[0].timer;

    // Implement LRU policy in this simulation
    // update timer
    LRU_timer(newset);

    // update hit counts when hit occurs
    for (int i = 0; i < E; i += 1) {
        if ((newset[i].valid == 1) && (newset[i].tag == tagbit)) {
            if (v == 1) {
                printf(" hit ");
            } // for the verbosity mode
            res.hits += 1;
            newset[i].timer = 0;
            // update counts of dirty bytes when write
            if ((operator== 'S') && (newset[i].dirty == 0)) {
                newset[i].dirty = 1;
                sum_dirty_bytes += B;
            }
            return;
        }
    }
    // update miss counts when it is not a hit
    res.misses += 1;
    if (v == 1) {
        printf(" miss ");
    }
    // check if there is an availble line
    for (int j = 0; j < E; j += 1) {
        if (newset[j].valid == 0) {
            newset[j].valid = 1;
            newset[j].tag = tagbit;
            if ((operator== 'S') && (newset[j].dirty == 0)) {
                newset[j].dirty = 1;
                sum_dirty_bytes += B;
            }
            return;
        }
    }

    // if there is no available line, evict using LRU replacement policy
    // find the index to the least recently used line
    for (int k = 0; k < E; k += 1) {
        if (newset[k].timer > LRU_base) {
            LRU_base = newset[k].timer;
            LRU_idx = k;
        }
    }

    // if LRU to be evicted is dirty, update counts of dirty bytes evicted
    if (newset[LRU_idx].dirty == 1) {
        res.dirty_evictions += B;
        newset[LRU_idx].dirty = 0; // reset dirty bit
    }

    // update eviction counts
    res.evictions += 1;
    // replace and update the least recently used line
    newset[LRU_idx].tag = tagbit;
    // reset the timer
    newset[LRU_idx].timer = 0;

    // update dirty bit and counts of dirty bytes
    for (int q = 0; q < E; q += 1) {
        if (newset[q].tag == tagbit) {
            if ((operator== 'S') && (newset[q].dirty == 0)) {
                newset[q].dirty = 1;
                sum_dirty_bytes += B;
            }
        }
    }

    if (v == 1) {
        printf("eviction ");
    } // verbosity mode
    return;
}

/* when command-line argument is h, print out the following usage information of
 * the cache */
void help_mode() {
    printf("Usage: ./csim-ref [-hv] -s <s> -E <E> -b <b> -t <tracefile>\n"
           "-h: Optional help flag that prints usage info\n"
           "-v: Optional verbose flag that displays trace info\n"
           "-s <s>: Number of set index bits (S = 2s is the number of sets)\n"
           "-E <E>: Associativity (number of lines per set)\n"
           "-b <b>: Number of block bits (B = 2b is the block size)\n"
           "-t <tracefile>: Name of the memory trace to replay\n");
}

int main(int argc, char *argv[]) {
    // initialize fields for the cache
    int input;
    cache newcache;

    // get command-line arguments
    while ((input = (getopt(argc, argv, "hvs:E:b:t:"))) != -1) {
        switch (input) {
        case 'h':
            h = 1; // print out usage information
            break;
        case 'v':
            v = 1; // verbosity mode;
            break;
        case 's':
            s = atoi(optarg);
            break;
        case 'E':
            E = atoi(optarg);
            break;
        case 'b':
            b = atoi(optarg);
            break;
        case 't':
            tracefile = (char *)optarg; // get tracefile name
            break;
        default:
            break;
        }
    } // done with getting command-line arguments

    /* if input is h, print out usage messages */
    if (h == 1) {
        help_mode();
        exit(0);
    } else {
        // check if inputs are valid
        if ((s < 0) || (E <= 0) || (b < 0) || (tracefile == NULL)) {
            printf("Invalid inputs!\n");
            exit(1);
        }
    }

    // initialize cache
    // S = 2^s
    S = 1 << s;
    // E = E
    // b = b, B = 2^b;
    B = 1 << b;

    // allocate memory for the cache
    newcache = (cache)malloc(sizeof(set) * S);
    // check if malloc is successful
    if (newcache == NULL) {
        printf("Allocation for sets failed.\n");
        exit(1);
    }
    for (int l = 0; l < S; l += 1) {
        newcache[l] = (set)malloc(sizeof(line) * E);
        // check if malloc is successful
        if (newcache[l] == NULL) {
            printf("Allcoation for lines failed.\n");
            exit(1);
        }
        // initialize fields for each line
        for (int k = 0; k < E; k += 1) {
            newcache[l][k].valid = 0;
            newcache[l][k].tag = 0;
            newcache[l][k].block = 0;
            newcache[l][k].dirty = 0;
            newcache[l][k].timer = 0;
        }
    } // done with cache initialization

    // cache operations and implementations-call helper function read_traces
    read_traces(newcache);

    // counts of dirty bytes = total dirty bytes written - dirty bytes evicted
    res.dirty_bytes = sum_dirty_bytes - res.dirty_evictions;
    printSummary(stats); // print out a summary of simulation statistics

    // free the allocated memory
    for (int f = 0; f < S; f += 1) {
        free(newcache[f]);
    }
    free(newcache); // use valgrind to check possible leaked memory - 0 errors
                    // reported

    return 0;
}
