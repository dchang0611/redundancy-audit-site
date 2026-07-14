# Daily Home Run Board

This project runs the home-run model every day on GitHub-hosted compute and publishes the top 30 picks as a mobile-friendly website.

## First-time setup

1. Create a GitHub repository and upload this folder.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, choose **GitHub Actions**.
4. Open **Actions → Refresh daily home run board → Run workflow** for the first run.

The workflow runs a preliminary board at 9:15 AM Pacific during daylight time, then refreshes at 11:45 AM and 2:15 PM as official lineups become available. GitHub schedules use UTC, so adjust the cron by one hour in winter if exact local timing matters.

Use the workflow's **Run workflow** button with an optional date for a manual rerun. The model automatically uses yesterday as the last complete Statcast data date and today as the board date.
