#include <stdlib.h> 

#include <drake/schedule.h> 

#define TASK_NAME task_1
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_1 1

#define TASK_NAME task_2
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_2 1

#define TASK_NAME task_3
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_3 1

#define TASK_NAME task_4
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_4 1

#define TASK_NAME task_5
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_5 1

#define TASK_NAME task_6
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_6 1

#define TASK_NAME task_7
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_7 1

#define TASK_NAME task_8
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_8 1

#define TASK_NAME task_9
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_9 1

#define TASK_NAME task_10
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_10 1

#define TASK_NAME task_11
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_11 1

#define TASK_NAME task_12
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_12 1

#define TASK_NAME task_13
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_13 1

#define TASK_NAME task_14
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_14 1

#define TASK_NAME task_15
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_task_15 1

int drake_task_number()
{
	return 15;
}

char* drake_task_name(size_t index)
{
	switch(index - 1)
	{
		case 0:
			return "task_1";
		break;
		case 1:
			return "task_2";
		break;
		case 2:
			return "task_3";
		break;
		case 3:
			return "task_4";
		break;
		case 4:
			return "task_5";
		break;
		case 5:
			return "task_6";
		break;
		case 6:
			return "task_7";
		break;
		case 7:
			return "task_8";
		break;
		case 8:
			return "task_9";
		break;
		case 9:
			return "task_10";
		break;
		case 10:
			return "task_11";
		break;
		case 11:
			return "task_12";
		break;
		case 12:
			return "task_13";
		break;
		case 13:
			return "task_14";
		break;
		case 14:
			return "task_15";
		break;
		default:
			return "invalid task id";
		break;
	}
}

void drake_schedule_init(drake_schedule_t* schedule)
{
	schedule->core_number = 1;
	schedule->task_number = 15;
	schedule->stage_time = 0;

	schedule->tasks_in_core = malloc(sizeof(size_t) * schedule->core_number);

	schedule->task_name = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->task_name[0] = "task_1";
	schedule->task_name[1] = "task_2";
	schedule->task_name[2] = "task_3";
	schedule->task_name[3] = "task_4";
	schedule->task_name[4] = "task_5";
	schedule->task_name[5] = "task_6";
	schedule->task_name[6] = "task_7";
	schedule->task_name[7] = "task_8";
	schedule->task_name[8] = "task_9";
	schedule->task_name[9] = "task_10";
	schedule->task_name[10] = "task_11";
	schedule->task_name[11] = "task_12";
	schedule->task_name[12] = "task_13";
	schedule->task_name[13] = "task_14";
	schedule->task_name[14] = "task_15";

	schedule->tasks_in_core[0] = 15;

	schedule->consumers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->consumers_in_core[0] = 0;

	schedule->producers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->producers_in_core[0] = 0;

	schedule->consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->consumers_in_task[0] = 0;
	schedule->consumers_in_task[1] = 1;
	schedule->consumers_in_task[2] = 1;
	schedule->consumers_in_task[3] = 1;
	schedule->consumers_in_task[4] = 1;
	schedule->consumers_in_task[5] = 1;
	schedule->consumers_in_task[6] = 1;
	schedule->consumers_in_task[7] = 1;
	schedule->consumers_in_task[8] = 1;
	schedule->consumers_in_task[9] = 1;
	schedule->consumers_in_task[10] = 1;
	schedule->consumers_in_task[11] = 1;
	schedule->consumers_in_task[12] = 1;
	schedule->consumers_in_task[13] = 1;
	schedule->consumers_in_task[14] = 1;

	schedule->producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->producers_in_task[0] = 2;
	schedule->producers_in_task[1] = 2;
	schedule->producers_in_task[2] = 2;
	schedule->producers_in_task[3] = 2;
	schedule->producers_in_task[4] = 2;
	schedule->producers_in_task[5] = 2;
	schedule->producers_in_task[6] = 2;
	schedule->producers_in_task[7] = 0;
	schedule->producers_in_task[8] = 0;
	schedule->producers_in_task[9] = 0;
	schedule->producers_in_task[10] = 0;
	schedule->producers_in_task[11] = 0;
	schedule->producers_in_task[12] = 0;
	schedule->producers_in_task[13] = 0;
	schedule->producers_in_task[14] = 0;

	schedule->remote_consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_consumers_in_task[0] = 0;
	schedule->remote_consumers_in_task[1] = 0;
	schedule->remote_consumers_in_task[2] = 0;
	schedule->remote_consumers_in_task[3] = 0;
	schedule->remote_consumers_in_task[4] = 0;
	schedule->remote_consumers_in_task[5] = 0;
	schedule->remote_consumers_in_task[6] = 0;
	schedule->remote_consumers_in_task[7] = 0;
	schedule->remote_consumers_in_task[8] = 0;
	schedule->remote_consumers_in_task[9] = 0;
	schedule->remote_consumers_in_task[10] = 0;
	schedule->remote_consumers_in_task[11] = 0;
	schedule->remote_consumers_in_task[12] = 0;
	schedule->remote_consumers_in_task[13] = 0;
	schedule->remote_consumers_in_task[14] = 0;

	schedule->remote_producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_producers_in_task[0] = 0;
	schedule->remote_producers_in_task[1] = 0;
	schedule->remote_producers_in_task[2] = 0;
	schedule->remote_producers_in_task[3] = 0;
	schedule->remote_producers_in_task[4] = 0;
	schedule->remote_producers_in_task[5] = 0;
	schedule->remote_producers_in_task[6] = 0;
	schedule->remote_producers_in_task[7] = 0;
	schedule->remote_producers_in_task[8] = 0;
	schedule->remote_producers_in_task[9] = 0;
	schedule->remote_producers_in_task[10] = 0;
	schedule->remote_producers_in_task[11] = 0;
	schedule->remote_producers_in_task[12] = 0;
	schedule->remote_producers_in_task[13] = 0;
	schedule->remote_producers_in_task[14] = 0;

	schedule->producers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->producers_id[0] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[0][0] = 2;
	schedule->producers_id[0][1] = 3;
	schedule->producers_id[1] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[1][0] = 4;
	schedule->producers_id[1][1] = 5;
	schedule->producers_id[2] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[2][0] = 7;
	schedule->producers_id[2][1] = 6;
	schedule->producers_id[3] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[3][0] = 9;
	schedule->producers_id[3][1] = 8;
	schedule->producers_id[4] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[4][0] = 11;
	schedule->producers_id[4][1] = 10;
	schedule->producers_id[5] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[5][0] = 12;
	schedule->producers_id[5][1] = 13;
	schedule->producers_id[6] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[6][0] = 15;
	schedule->producers_id[6][1] = 14;
	schedule->producers_id[7] = NULL;
	schedule->producers_id[8] = NULL;
	schedule->producers_id[9] = NULL;
	schedule->producers_id[10] = NULL;
	schedule->producers_id[11] = NULL;
	schedule->producers_id[12] = NULL;
	schedule->producers_id[13] = NULL;
	schedule->producers_id[14] = NULL;

	schedule->consumers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->consumers_id[0] = NULL;
	schedule->consumers_id[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[1][0] = 1;
	schedule->consumers_id[2] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[2][0] = 1;
	schedule->consumers_id[3] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[3][0] = 2;
	schedule->consumers_id[4] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[4][0] = 2;
	schedule->consumers_id[5] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[5][0] = 3;
	schedule->consumers_id[6] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[6][0] = 3;
	schedule->consumers_id[7] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[7][0] = 4;
	schedule->consumers_id[8] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[8][0] = 4;
	schedule->consumers_id[9] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[9][0] = 5;
	schedule->consumers_id[10] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[10][0] = 5;
	schedule->consumers_id[11] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[11][0] = 6;
	schedule->consumers_id[12] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[12][0] = 6;
	schedule->consumers_id[13] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[13][0] = 7;
	schedule->consumers_id[14] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[14][0] = 7;

	schedule->schedule = malloc(sizeof(drake_schedule_task_t*) * schedule->core_number);
	schedule->schedule[0] = malloc(sizeof(drake_schedule_task_t) * 15);
	schedule->schedule[0][0].id = 1;
	schedule->schedule[0][0].start_time = 0;
	schedule->schedule[0][0].frequency = 1;
	schedule->schedule[0][1].id = 3;
	schedule->schedule[0][1].start_time = 32;
	schedule->schedule[0][1].frequency = 1;
	schedule->schedule[0][2].id = 2;
	schedule->schedule[0][2].start_time = 48;
	schedule->schedule[0][2].frequency = 1;
	schedule->schedule[0][3].id = 7;
	schedule->schedule[0][3].start_time = 64;
	schedule->schedule[0][3].frequency = 1;
	schedule->schedule[0][4].id = 6;
	schedule->schedule[0][4].start_time = 72;
	schedule->schedule[0][4].frequency = 1;
	schedule->schedule[0][5].id = 5;
	schedule->schedule[0][5].start_time = 80;
	schedule->schedule[0][5].frequency = 1;
	schedule->schedule[0][6].id = 4;
	schedule->schedule[0][6].start_time = 88;
	schedule->schedule[0][6].frequency = 1;
	schedule->schedule[0][7].id = 13;
	schedule->schedule[0][7].start_time = 96;
	schedule->schedule[0][7].frequency = 1;
	schedule->schedule[0][8].id = 12;
	schedule->schedule[0][8].start_time = 100;
	schedule->schedule[0][8].frequency = 1;
	schedule->schedule[0][9].id = 11;
	schedule->schedule[0][9].start_time = 104;
	schedule->schedule[0][9].frequency = 1;
	schedule->schedule[0][10].id = 10;
	schedule->schedule[0][10].start_time = 108;
	schedule->schedule[0][10].frequency = 1;
	schedule->schedule[0][11].id = 9;
	schedule->schedule[0][11].start_time = 112;
	schedule->schedule[0][11].frequency = 1;
	schedule->schedule[0][12].id = 8;
	schedule->schedule[0][12].start_time = 116;
	schedule->schedule[0][12].frequency = 1;
	schedule->schedule[0][13].id = 15;
	schedule->schedule[0][13].start_time = 120;
	schedule->schedule[0][13].frequency = 1;
	schedule->schedule[0][14].id = 14;
	schedule->schedule[0][14].start_time = 122;
	schedule->schedule[0][14].frequency = 1;
}

void drake_schedule_destroy(drake_schedule_t* schedule)
{
	free(schedule->schedule[0]);

	free(schedule->schedule);
	free(schedule->consumers_id[0]);
	free(schedule->consumers_id[1]);
	free(schedule->consumers_id[2]);
	free(schedule->consumers_id[3]);
	free(schedule->consumers_id[4]);
	free(schedule->consumers_id[5]);
	free(schedule->consumers_id[6]);
	free(schedule->consumers_id[7]);
	free(schedule->consumers_id[8]);
	free(schedule->consumers_id[9]);
	free(schedule->consumers_id[10]);
	free(schedule->consumers_id[11]);
	free(schedule->consumers_id[12]);
	free(schedule->consumers_id[13]);
	free(schedule->consumers_id[14]);
	free(schedule->consumers_id);

	free(schedule->producers_id[0]);
	free(schedule->producers_id[1]);
	free(schedule->producers_id[2]);
	free(schedule->producers_id[3]);
	free(schedule->producers_id[4]);
	free(schedule->producers_id[5]);
	free(schedule->producers_id[6]);
	free(schedule->producers_id[7]);
	free(schedule->producers_id[8]);
	free(schedule->producers_id[9]);
	free(schedule->producers_id[10]);
	free(schedule->producers_id[11]);
	free(schedule->producers_id[12]);
	free(schedule->producers_id[13]);
	free(schedule->producers_id[14]);
	free(schedule->producers_id);
	free(schedule->remote_producers_in_task);
	free(schedule->remote_consumers_in_task);
	free(schedule->producers_in_task);
	free(schedule->consumers_in_task);
	free(schedule->producers_in_core);
	free(schedule->consumers_in_core);
	free(schedule->tasks_in_core);
	free(schedule->task_name);
}

void*
drake_function(size_t id, task_status_t status)
{
	switch(id)
	{
		default:
			// TODO: Raise an alert
		break;
		case 1:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_1);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_1);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_1);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_1);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_1);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 2:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_2);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_2);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_2);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_2);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_2);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 3:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_3);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_3);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_3);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_3);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_3);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 4:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_4);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_4);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_4);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_4);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_4);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 5:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_5);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_5);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_5);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_5);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_5);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 6:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_6);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_6);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_6);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_6);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_6);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 7:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_7);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_7);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_7);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_7);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_7);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 8:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_8);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_8);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_8);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_8);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_8);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 9:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_9);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_9);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_9);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_9);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_9);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 10:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_10);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_10);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_10);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_10);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_10);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 11:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_11);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_11);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_11);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_11);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_11);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 12:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_12);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_12);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_12);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_12);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_12);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 13:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_13);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_13);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_13);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_13);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_13);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 14:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_14);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_14);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_14);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_14);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_14);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 15:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, task_15);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, task_15);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, task_15);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, task_15);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, task_15);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;

	}

	return 0;
}
