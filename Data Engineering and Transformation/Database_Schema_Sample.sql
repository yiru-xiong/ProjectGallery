CREATE TABLE "user_details" (
  "USER_ID" int PRIMARY KEY,
  "CREATED_AT" timestamp,
  "FIRST_BACKGROUND_CHECK_PASSED_AT" timestamp,
  "FIRST_JOB_ACCEPTED_AT" timestamp,
  "FIRST_JOB_APPROVED_AT" timestamp,
  "LAST_JOB_APPROVED_AT" timestamp,
  "SUSPENDED" boolean,
  "BADGE_NAMES" varchar,
  "ZIP" int,
  "GENDER" varchar
);

CREATE TABLE "client_details" (
  "CLIENT_ID" int PRIMARY KEY,
  "PARENT_ID" varchar,
  "CREATED_DATE" timestamp,
  "ACTIVATED_DATE" timestamp
);


CREATE TABLE "job_requests" (
  "JOB_REQUEST_ID" int PRIMARY KEY,
  "CLIENT_ID" int,
  "STATE" varchar,
  "CREATED_AT" timestamp,
  "POSTED_AT" timestamp,
  "ZIP" int,
  "SLOTS" int,
  "CATEGORY" varchar,
  "HOURS" float,
  "PAY_RATE" float,
  "START_AT" timestamp,
  "END_AT" timestamp,
  "REQUEST_NAME" varchar,
  "CANCELLED_AT" timestamp,
  "BADGE_NAMES" varchar
);

CREATE TABLE "jobs" (
  "JOB_ID" int,
  "JOB_REQUEST_ID" int,
  "USER_ID" int,
  "STATE" varchar,
  "STARTED_AT" timestamp,
  "COMPLETED_AT" timestamp,
  "CANCELLED_AT" timestamp,
  "WITHDRAWN_AT" timestamp,
  "POSTED_PAY" float,
  "FINAL_PAY" float,
  "BONUS_PAY" float,
  "BAN" boolean,
  "AVG_RATING" float,
  "AVG_CLIENT_RATING" float
);

CREATE TABLE "user_activity" (
  "USER_ID" int,
  "DATE" timestamp,
  "PUSH_NOTIFS_ON" boolean,
  "NUM_SESSIONS" int,
  "AVG_SESSION_TIME" float,
  "FIND_JOBS_TAPS" int,
  "NUM_JOBS_VIEWED" int,
  "NUM_JOBS_ACCEPTED" int,
  "NUM_NOTIFS" int
);

CREATE TABLE "job_request_tasks" (
  "JOB_REQUEST_ID" int,
  "DESCRIPTION_SKILLS" varchar,
  "DESCRIPTION_TASKS" varchar
);


CREATE TABLE "client_ratings" (
  "CLIENT_ID" int,
  "USER_ID" int,
  "JOB_ID" int,
  "RATING_OF_CLIENT" int,
  "RATING_OF_CLIENT_COMMENT" varchar
);


ALTER TABLE "user_activity" ADD FOREIGN KEY ("USER_ID") REFERENCES "user_details" ("USER_ID");

ALTER TABLE "jobs" ADD FOREIGN KEY ("USER_ID") REFERENCES "user_details" ("USER_ID");

ALTER TABLE "job_requests" ADD FOREIGN KEY ("CLIENT_ID") REFERENCES "client_details" ("CLIENT_ID");

ALTER TABLE "client_details" ADD FOREIGN KEY ("PARENT_ID") REFERENCES "parent_companies" ("PARENT_ID");

ALTER TABLE "jobs" ADD FOREIGN KEY ("JOB_REQUEST_ID") REFERENCES "job_requests" ("JOB_REQUEST_ID");

ALTER TABLE "job_request_tasks" ADD FOREIGN KEY ("JOB_REQUEST_ID") REFERENCES "job_requests" ("JOB_REQUEST_ID");

ALTER TABLE "jobs" ADD FOREIGN KEY ("JOB_ID") REFERENCES "client_ratings" ("JOB_ID");

ALTER TABLE "jobs" ADD FOREIGN KEY ("JOB_ID") REFERENCES "ratings" ("JOB_ID");

ALTER TABLE "client_details" ADD FOREIGN KEY ("CLIENT_ID") REFERENCES "ratings" ("CLIENT_ID");

ALTER TABLE "user_details" ADD FOREIGN KEY ("USER_ID") REFERENCES "client_ratings" ("USER_ID");

ALTER TABLE "client_ratings" ADD FOREIGN KEY ("CLIENT_ID") REFERENCES "client_details" ("CLIENT_ID");

