"""Self-serve notebook authoring tools: validate and annotate jobs.

store     the job queue (jobs/jobs.db) and each job's directory
lint      contribution-guideline checks that need no GPU
capacity  which runner.yaml target has a free slot, and runner availability
pipeline  the stages of one job (run inside a job container)
worker    the scheduler: starts jobs as slots free up
app       the web UI where authors submit jobs and follow them
"""
