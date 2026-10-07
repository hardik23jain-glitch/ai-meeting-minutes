# Meeting Record

## Summary
The team reviewed the payments project backend migration, noting API latency at 450 ms versus a target under 200 ms due to PostgreSQL queries. A decision was made to add a Redis cache and remain with PostgreSQL, while a Kubernetes migration was only suggested. The budget of $5,000 for cloud costs was approved, and several action items were assigned. The next meeting will be at the same time next week.

## Minutes
### API latency
- API latency currently about 450 milliseconds, target under 200 milliseconds
- Main cause identified as PostgreSQL queries on the orders table

### Kubernetes suggestion
- Suggestion to move the whole system to Kubernetes was mentioned but not agreed

### Redis cache decision
- Decision to add Redis cache in front of the orders service

### Database choice
- Decision to stay with PostgreSQL and not use MongoDB

### Budget
- Approved five thousand dollars for cloud cost this month

### CI/CD pipeline issue
- Builds are failing on GitHub Actions; someone needs to look at the CI/CD pipeline, not yet assigned

### Next meeting
- Same time next week

## Key Decisions
- Add Redis cache in front of the orders service
- Do not use MongoDB; stay with PostgreSQL
- Approve five thousand dollars for cloud cost this month

## Action Items
- Set up the Redis cache (Owner: Rahul; Deadline: Friday the 10th of October)
- Write the load test using Locust (Owner: Priya; Deadline: unspecified)
- Look at the CI/CD pipeline because builds keep failing on GitHub Actions (Owner: unspecified; Deadline: unspecified)

## Open Items (discussed, not decided)
- Consider moving the whole system to Kubernetes: Suggested but not agreed or decided