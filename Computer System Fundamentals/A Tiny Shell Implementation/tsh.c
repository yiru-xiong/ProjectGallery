/**
 * @file tsh.c
 * @brief A tiny shell program with job control
 * This tiny shell is an interactive command-line interpreter that repeatedly
 * prints prompt, waits for a command line on stdin, interprets contents of
 * command lines, creates child process to run executables and reaps the child
 * when it dies.
 * Valid commands could be a path to an executable file or built-in commands
 * including quit, jobs, bg job and fg job. This shell supports both foreground
 * jobs and background jobs (command line ends with a '&' character) and can
 * switch between two status.
 * When user types in Ctrl-C or Ctrl-Z, the shell will catch the corresponding
 * signals and forward them to process group with foreground jobs.
 * By default,the shell reads input from the keybaord and displays output to its
 * terminal. Input and output (I/O) could also be redirected from/to another
 * locations using proper '<' or '>' characters.
 * @author Yiru Xiong
 */

#include "csapp.h"
#include "tsh_helper.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/*
 * If DEBUG is defined, enable contracts and printing on dbg_printf.
 */
#ifdef DEBUG
/* When debugging is enabled, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(...) assert(__VA_ARGS__)
#define dbg_assert(...) assert(__VA_ARGS__)
#define dbg_ensures(...) assert(__VA_ARGS__)
#else
/* When debugging is disabled, no code gets generated for these */
#define dbg_printf(...)
#define dbg_requires(...)
#define dbg_assert(...)
#define dbg_ensures(...)
#endif

/** @brief Add access mode and file creation flags **/
/* ORDONLY: opening the file read-only
** OWRONLY: opening the file write-only
** O_CREAT: create a regular file if pathname does not exist
** O_TRUNC: if the file already exists and the access mode allows writing,
** it will be truncated to length 0
**/
#define IN_ACCESS O_RDONLY
#define OUT_ACCESS O_CREAT | O_TRUNC | O_WRONLY

/** @brief Add file mode bits **/
/* S_IRUSR: set read rights for the owner
** S_IWUSR : set write rights for the owner
** S_IRGRP: set read rights for the group
** S_IROTH: set read rights for other users
**/
#define MODE S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH

/* Function prototypes */
void eval(const char *cmdline); // modified by YX
void builtin_commands(struct cmdline_tokens token, sigset_t mask,
                      sigset_t prev);             // added by YX
void IO_redirection(struct cmdline_tokens token); // added by YX
void IO_error_handling(char *filename);           // added by YX
void sigchld_handler(int sig);                    // modified by YX
void sigtstp_handler(int sig);                    // modified by YX
void sigint_handler(int sig);                     // modified by YX
void sigquit_handler(int sig);
void cleanup(void);

/**
 * @brief the main routine of the shell
 * This function sets up environment variables, parse command line options,
 * initialize job list, set up signal handlers, read in commands and invoke eval
 * function to execute them
 * @param argc argument count
 * @param argv argument vector
 * @return integer indicates exist status of the program (if it exists normally)
 */
int main(int argc, char **argv) {
    char c;
    char cmdline[MAXLINE_TSH]; // Cmdline for fgets
    bool emit_prompt = true;   // Emit prompt (default)

    // Redirect stderr to stdout (so that driver will get all output
    // on the pipe connected to stdout)
    if (dup2(STDOUT_FILENO, STDERR_FILENO) < 0) {
        perror("dup2 error");
        exit(1);
    }

    // Parse the command line
    while ((c = getopt(argc, argv, "hvp")) != EOF) {
        switch (c) {
        case 'h': // Prints help message
            usage();
            break;
        case 'v': // Emits additional diagnostic info
            verbose = true;
            break;
        case 'p': // Disables prompt printing
            emit_prompt = false;
            break;
        default:
            usage();
        }
    }

    // Create environment variable
    if (putenv("MY_ENV=42") < 0) {
        perror("putenv error");
        exit(1);
    }

    // Set buffering mode of stdout to line buffering.
    // This prevents lines from being printed in the wrong order.
    if (setvbuf(stdout, NULL, _IOLBF, 0) < 0) {
        perror("setvbuf error");
        exit(1);
    }

    // Initialize the job list
    init_job_list();

    // Register a function to clean up the job list on program termination.
    // The function may not run in the case of abnormal termination (e.g. when
    // using exit or terminating due to a signal handler), so in those cases,
    // we trust that the OS will clean up any remaining resources.
    if (atexit(cleanup) < 0) {
        perror("atexit error");
        exit(1);
    }

    // Install the signal handlers
    Signal(SIGINT, sigint_handler);   // Handles Ctrl-C
    Signal(SIGTSTP, sigtstp_handler); // Handles Ctrl-Z
    Signal(SIGCHLD, sigchld_handler); // Handles terminated or stopped child

    Signal(SIGTTIN, SIG_IGN);
    Signal(SIGTTOU, SIG_IGN);

    Signal(SIGQUIT, sigquit_handler);

    // Execute the shell's read/eval loop
    while (true) {
        if (emit_prompt) {
            printf("%s", prompt);

            // We must flush stdout since we are not printing a full line.
            fflush(stdout);
        }

        if ((fgets(cmdline, MAXLINE_TSH, stdin) == NULL) && ferror(stdin)) {
            perror("fgets error");
            exit(1);
        }

        if (feof(stdin)) {
            // End of file (Ctrl-D)
            printf("\n");
            return 0;
        }

        // Remove any trailing newline
        char *newline = strchr(cmdline, '\n');
        if (newline != NULL) {
            *newline = '\0';
        }

        // Evaluate the command line
        eval(cmdline);
    }

    return -1; // control never reaches here
}

/**
 * @brief evaluate and execute command lines
 * Built-in commands (quit, jobs, bg job, fg job) runs within the shell's
 * process and will be executed immediately Non built-in commands will be
 * executed through forking a child process and run in its child process Jobs
 * could be run in foreground or background and their status could be switched
 * I/O redirection is also implemented and enables user to redirect stdin/stdout
 * from/to other locations If errors (permission/file errors etc.) are
 * encountered during the process, corresponding error messages will be printed
 * NOTE: The shell is supposed to be a long-running process, so this function
 *       (and its helpers) should avoid exiting on error.  This is not to say
 *       they shouldn't detect and print (or otherwise handle) errors!
 * @param[in] cmdline the command line
 * @return
 */
void eval(const char *cmdline) {
    // initialize fields
    parseline_return parse_result; // Parseline results: ERROR, EMPTY, FG, BG
    struct cmdline_tokens token;   // struct holding results from parseline
    pid_t pid;                     // Process ID
    jid_t jid;                     // Job ID (with prefix %)

    // set up for signal handling
    sigset_t mask, prev;
    sigfillset(&mask);
    sigemptyset(&mask);
    sigaddset(&mask, SIGCHLD);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTSTP);
    sigaddset(&mask, SIGTERM);

    // Parse command line
    parse_result = parseline(cmdline, &token);

    // Ignore empty line
    if (token.argv[0] == NULL) {
        return;
    }

    // Ignore if parseline result is empty or erroneous
    if (parse_result == PARSELINE_ERROR || parse_result == PARSELINE_EMPTY) {
        return;
    }

    // Check if a command is built-in and perform accordingly
    if (token.builtin != BUILTIN_NONE) {
        // run if a command is built-in: quit, jobs, bg job and fg job
        builtin_commands(token, mask, prev);
        // command is not built-in
    } else {
        sigprocmask(SIG_BLOCK, &mask, &prev);
        /* child process */
        // create a new process using fork
        if ((pid = fork()) == 0) {
            sigprocmask(SIG_SETMASK, &prev, NULL);
            // implemented I/O redirection - able to redirect input/output
            // from/to locations other than default
            IO_redirection(token);
            setpgid(0, 0);
            if (execve(token.argv[0], token.argv, environ) < 0) {
                // file and permission errors handling
                if (errno == ENOENT) {
                    perror(token.argv[0]);
                    exit(1);
                }
                if (errno == EACCES) {
                    perror(token.argv[0]);
                    exit(1);
                }
                // IO_error_handling(token.argv[0]);
                _exit(0);
            }
            /* Parent Process */
        } else {
            // check for fork failures
            if (pid < 0) {
                perror("Error: failed to fork a child process\n");
                return;
            }

            sigprocmask(SIG_BLOCK, &mask, NULL);
            // job runs on foreground
            if (parse_result == PARSELINE_FG) {
                add_job(pid, FG, cmdline); // add the child to job list and
                                           // udpate the state to FG
                // wait for a child process to end if running a foreground job
                while (fg_job() != 0) {
                    sigsuspend(&prev); // waiting for signals
                }
            }
            // job runs on background (command line ends with a &)
            if (parse_result == PARSELINE_BG) {
                jid = add_job(pid, BG, cmdline);
                // print out command line, prepend with JID and PID as requested
                // for background jobs
                sio_printf("[%d] (%d) %s\n", jid, pid, job_get_cmdline(jid));
            }
            sigprocmask(SIG_SETMASK, &prev, NULL);
        }
    }
    return;
}

/**************** Helper Functions Added by YX ****************/
/**
 * @brief run built-in commands
 * supported commands include quit, jobs, bg job and fg job
 * @param[in] cmdline_tokens token. Struct holds information of results from
 * parseline
 * @param[in] mask, prev. Signal set
 * @return
 */
void builtin_commands(struct cmdline_tokens token, sigset_t mask,
                      sigset_t prev) {
    pid_t pid; // Process ID
    jid_t jid; // Job ID (with prefix %)

    // for I/O redirection
    int rio_out;
    rio_out = STDOUT_FILENO;
    sigprocmask(SIG_BLOCK, &mask, &prev); // block signals
    if (token.builtin == BUILTIN_QUIT) {
        exit(0);
    } else if (token.builtin == BUILTIN_JOBS) {
        rio_out = STDOUT_FILENO;
        // jobs command: list all background jobs
        if (token.outfile != NULL) {
            // Implement I/O redirection when outfile is not NULL
            rio_out = open(token.outfile, OUT_ACCESS, MODE);
            if (rio_out > 0) {
                list_jobs(rio_out);
            }
            // handling file and permission errors
            if (errno == ENOENT) {
                perror(token.outfile);
            }
            if (errno == EACCES) {
                perror(token.outfile);
            }
            // close the file
            if (rio_out != STDOUT_FILENO) {
                close(rio_out);
            }
        } else {
            list_jobs(STDOUT_FILENO); // default output
        }
        sigprocmask(SIG_SETMASK, &prev, NULL); 
        // bg job & fg job commands: resume jobs by sending a SIGCONT and run in
        // foreground or background as requested
    } else if ((token.builtin == BUILTIN_FG) || (token.builtin == BUILTIN_BG)) {
        // job argument can be either PID or JID.
        // check for NULL inputs and invalid inputs
        if (token.argv[1] == NULL) {
            sigprocmask(SIG_SETMASK, &prev, NULL);
            sio_printf("%s: argument must be a PID or %%jobid\n",
                       token.argv[0]);
            return;
        }
        // JID is identified by prefix %
        if (token.argv[1][0] == '%') {
            jid = atoi(token.argv[1] + sizeof(char));
            if (job_exists(jid) == false) {
                sigprocmask(SIG_SETMASK, &prev, NULL);
                sio_printf("%s: No such job\n", token.argv[1]);
                return;
            }
        } else if (isdigit(token.argv[1][0])) {
            pid = atoi(token.argv[1]);
            jid = job_from_pid(pid);
            if (job_exists(jid) == false) {
                sigprocmask(SIG_SETMASK, &prev, NULL);
                sio_printf("%s: No such job\n", token.argv[1]);
                return;
            }
        } else {
            sigprocmask(SIG_SETMASK, &prev, NULL);
            sio_printf("%s command requires PID or %%jobid argument\n",
                       token.argv[0]);
            return;
        }
        // get process ID of a job by JID
        pid = job_get_pid(jid);
        if (token.builtin == BUILTIN_FG) {
            kill(-pid, SIGCONT);
            job_set_state(jid, FG);
            while (fg_job() != 0) {
                sigsuspend(&prev);
            }
            sigprocmask(SIG_SETMASK, &prev, NULL);
        } else if (token.builtin == BUILTIN_BG) {
            kill(-pid, SIGCONT);
            if (job_exists(jid)) {
                job_set_state(jid, BG);
                sio_printf("[%d] (%d) %s\n", jid, pid, job_get_cmdline(jid));
            } else {
                sio_printf("[%d]: No such job\n", jid);
            }
            sigprocmask(SIG_SETMASK, &prev, NULL);
        }
    }
    return;
}

/**
 * @brief Enable the shell to support input/output redirection
 * change the default input source to the input infile prepended by < in command
 * line redirect the default output to requested destination prepended by > in
 * command line Access modes and file mode bits are defined in macros
 * @param[in] cmdline_tokens token Struct holds information of results from
 * parseline
 * @return
 */
void IO_redirection(struct cmdline_tokens token) {

    int rio_in, rio_out;
    rio_in = STDIN_FILENO;   // initialize to stdin
    rio_out = STDOUT_FILENO; // initialize to stdout

    // redirect stdin when infile exists
    if (token.infile != NULL) {
        rio_in = open(token.infile, IN_ACCESS, MODE);
        // handling file error or permission errors
        IO_error_handling(token.infile);

        if (rio_in > 0) {
            dup2(rio_in, STDIN_FILENO);
        }

        // close the file
        if (rio_in != STDIN_FILENO) {
            close(rio_in);
        }
    }
    // redirect stout when outfile exits
    if (token.outfile != NULL) {
        rio_out = open(token.outfile, OUT_ACCESS, MODE);
        // handling file error or permission errors
        IO_error_handling(token.outfile);

        if (rio_out > 0) {
            dup2(rio_out, STDOUT_FILENO);
        }

        // close the file
        if (rio_out != STDOUT_FILENO) {
            close(rio_out);
        }
    }
}

/**
 * @brief Identify and report file error and permission error in I/O redirection
 * @param[in] filename File name for input or output
 * @return
 */
void IO_error_handling(char *filename) {
    // error occurs when there is no such file or directory
    if (errno == ENOENT) {
        perror(filename);
        exit(1);
    }
    // error occurs when the permission is denied
    if (errno == EACCES) {
        perror(filename);
        exit(1);
    }
}

/*****************
 * Signal handlers
 *****************/
/**
 * @brief Handle SIGCHLD signal sent by kernel to the shell
 * SIGCHLD signal will be sent when a child process terminates and becomes a
 * zombie or the process stops due to SIGTSTP or SIGINT signals. It reaps zombie
 * child
 * @param[in] sig
 * @return
 */
void sigchld_handler(int sig) {
    pid_t pid;
    jid_t jid;
    int status;
    // save errno to prevent corruption
    int oldererr;
    oldererr = errno;

    sigset_t mask, prev;
    sigfillset(&mask);
    sigemptyset(&mask);
    sigaddset(&mask, SIGCHLD);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTSTP);
    sigaddset(&mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &mask, &prev);
    // WNOHANG: waitpid should return immediately instead of waiting
    // WUNTRACED: report status of termianted or stopped child process
    while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {

        jid = job_from_pid(pid);
        // handle normally terminated casses
        if (WIFEXITED(status)) {
            sigprocmask(SIG_BLOCK, &mask, &prev);
            delete_job(jid); // delete a job by jid from the job list
            sigprocmask(SIG_SETMASK, &prev, NULL);
            // handle signals like Ctrl-C
        } else if (WIFSIGNALED(status)) {
            sigprocmask(SIG_BLOCK, &mask, &prev);
            sio_printf("Job [%d] (%d) terminated by signal %d\n", jid, pid,
                       WTERMSIG(status));
            delete_job(jid);
            sigprocmask(SIG_SETMASK, &prev, NULL);
            // handle stop signals like Ctrl-Z
        } else {
            sigprocmask(SIG_BLOCK, &mask, &prev);
            sio_printf("Job [%d] (%d) stopped by signal %d\n", jid, pid,
                       WSTOPSIG(status));
            job_set_state(jid, ST);
            sigprocmask(SIG_SETMASK, &prev, NULL);
        }
    }
    // ECHILD: no child process
    // check for errors
    if (pid < 0 && errno != ECHILD) {
        perror("Error: waitpid error!\n");
    }
    errno = oldererr; // restore errno
    return;
}

/**
 * @brief Handle SIGINT signal when user types in Ctrl-C
 * SIGINT - interrupt signal, default action is to terminate the process
 * The shell catches the signal and forward it to the entire process group with
 * foreground job This signal will have no effect if there is no foreground job
 * @param[in] sig
 * @return
 */
void sigint_handler(int sig) {

    pid_t pid;
    jid_t jid;
    int olderrno;
    olderrno = errno; // save errno

    sigset_t mask, prev;
    sigemptyset(&mask);
    sigfillset(&mask);
    sigprocmask(SIG_BLOCK, &mask, &prev);

    jid = fg_job(); // get job id for current foregound job
    if (jid != 0) {
        pid = job_get_pid(jid);
        if (kill(-pid, SIGINT) == -1) {
            perror("Error: failed to kill the foreground job\n");
        }
    }
    sigprocmask(SIG_SETMASK, &prev, NULL);
    errno = olderrno; // restore errno
    return;
}

/**
 * @brief Handle SIGTSTP signal when user types in Ctrl-Z
 * SIGTSTP - terminal stop causes the process to suspend execution by default.
 * It could be handled or ignored The shell catches the signal and forward it to
 * the entire process group with foreground job This signal will have no effect
 * if there is no foreground job
 * @param[in] sig
 * @return
 */
void sigtstp_handler(int sig) {

    pid_t pid;
    jid_t jid;
    int olderrno;
    olderrno = errno; // save errno

    sigset_t mask, prev;
    sigemptyset(&mask);
    sigfillset(&mask);
    sigprocmask(SIG_BLOCK, &mask, &prev);

    jid = fg_job(); // get job id of current foreground job
    if (jid != 0) {
        pid = job_get_pid(jid);
        if (kill(-pid, SIGTSTP) == -1) {
            perror("Error: failed to kill the foreground job\n");
        }
    }
    sigprocmask(SIG_SETMASK, &prev, NULL);
    errno = olderrno; // restore errno
    return;
}

/**
 * @brief Attempt to clean up global resources when the program exits.
 *
 * In particular, the job list must be freed at this time, since it may
 * contain leftover buffers from existing or even deleted jobs.
 */
void cleanup(void) {
    // Signals handlers need to be removed before destroying the joblist
    Signal(SIGINT, SIG_DFL);  // Handles Ctrl-C
    Signal(SIGTSTP, SIG_DFL); // Handles Ctrl-Z
    Signal(SIGCHLD, SIG_DFL); // Handles terminated or stopped child

    destroy_job_list();
}
