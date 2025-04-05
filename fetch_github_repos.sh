#!/bin/bash
# 1. Configuration and state file initialization
GITHUB_API="https://api.github.com"           # [Line 2]
PER_PAGE=100                                  # [Line 3]
STATE_FILE="process_state.txt"                # [Line 4]
BASE_DIR="/Users/hananiah"                               # [Line 5]
REPOS_DIR="$BASE_DIR/repos"                   # [Line 6]
READMES_DIR="$BASE_DIR/READMEs"               # [Line 7]

mkdir -p "$REPOS_DIR"                         # [Line 9]: Ensure the repos directory exists
mkdir -p "$READMES_DIR"                       # [Line 10]: Ensure the READMEs directory exists

# 2. Load or initialize state variables from the state file
if [ -f "$STATE_FILE" ]; then                 # [Line 13]
    source "$STATE_FILE"                      # [Line 14]: Load SINCE, CURRENT_USER, CURRENT_REPO_INDEX
else
    SINCE=0                                   # [Line 16]
    CURRENT_USER=""                           # [Line 17]
    CURRENT_REPO_INDEX=0                      # [Line 18]
fi

# 3. Function to update the state file
update_state() {                               # [Line 21]
    echo "SINCE=$SINCE" > "$STATE_FILE"       # [Line 22]
    echo "CURRENT_USER=$CURRENT_USER" >> "$STATE_FILE"  # [Line 23]
    echo "CURRENT_REPO_INDEX=$CURRENT_REPO_INDEX" >> "$STATE_FILE"  # [Line 24]
}

# 4. Function to fetch users and process their repositories
fetch_users() {                                # [Line 27]
    while true; do                           # [Line 28]
        echo "Fetching users since ID: $SINCE..."
        USERS_JSON=$(curl -s "$GITHUB_API/users?since=$SINCE&per_page=$PER_PAGE")  # [Line 30]
        LOGINS=($(echo "$USERS_JSON" | jq -r '.[].login'))  # [Line 31]
        LAST_ID=$(echo "$USERS_JSON" | jq -r '.[-1].id')      # [Line 32]

        if [ -z "$LAST_ID" ] || [ "${#LOGINS[@]}" -eq 0 ]; then  # [Line 34]
            echo "No more users found. Exiting."
            break
        fi

        echo "Last User ID: $LAST_ID"       # [Line 39]

        # Process each user in the batch
        for LOGIN in "${LOGINS[@]}"; do     # [Line 42]
            # If resuming from a previous run, only process the saved CURRENT_USER
            if [ -n "$CURRENT_USER" ] && [ "$CURRENT_USER" != "$LOGIN" ]; then  # [Line 44]
                echo "Skipping user $LOGIN as current state is for $CURRENT_USER"
                continue
            fi

            # If no current user is stored, set the current user to the one being processed now
            if [ -z "$CURRENT_USER" ]; then  # [Line 49]
                CURRENT_USER="$LOGIN"        # [Line 50]
                CURRENT_REPO_INDEX=0         # [Line 51]
                update_state                 # [Line 52]
            fi

            echo "Processing user: $LOGIN"   # [Line 55]
            REPOS_JSON=$(curl -s "$GITHUB_API/users/$LOGIN/repos?per_page=100")  # [Line 56]
            REPO_URLS=($(echo "$REPOS_JSON" | jq -r '.[].clone_url'))  # [Line 57]
            REPO_NAMES=($(echo "$REPOS_JSON" | jq -r '.[].name'))       # [Line 58]

            # Process each repository for the current user
            for i in "${!REPO_URLS[@]}"; do  # [Line 61]
                # Skip repos that were fully processed in a previous run
                if [ "$i" -lt "$CURRENT_REPO_INDEX" ]; then  # [Line 63]
                    continue
                fi

                REPO_URL="${REPO_URLS[$i]}"    # [Line 66]
                REPO_NAME="${REPO_NAMES[$i]}"  # [Line 67]
                REPO_PATH="$REPOS_DIR/$REPO_NAME"  # [Line 68]

                echo "Cloning $REPO_NAME..."  # [Line 70]
                git clone --depth=1 "$REPO_URL" "$REPO_PATH"  # [Line 71]

                # Check multiple locations for README.md:
                README_PATH=""
                if [ -f "$REPO_PATH/README.md" ]; then             # [Line 75]
                    README_PATH="$REPO_PATH/README.md"
                elif [ -f "$REPO_PATH/docs/README.md" ]; then         # [Line 77]
                    README_PATH="$REPO_PATH/docs/README.md"
                elif [ -f "$REPO_PATH/.github/README.md" ]; then       # [Line 79]
                    README_PATH="$REPO_PATH/.github/README.md"
                fi
                
                if [ -n "$README_PATH" ]; then                        # [Line 82]
                    cp "$README_PATH" "$READMES_DIR/$REPO_NAME.md"  # [Line 83]
                    zip -r "$REPO_PATH.zip" "$REPO_PATH"              # [Line 84]
                    rm -rf "$REPO_PATH"                               # [Line 85]
                else                                                  # [Line 86]
                    echo "No README.md found in $REPO_NAME (searched root, docs, .github); deleting repo."  # [Line 87]
                    rm -rf "$REPO_PATH"                               # [Line 88]
                fi

                # Update the current repo index in state (repo will be reprocessed if script stops here)
                CURRENT_REPO_INDEX=$((i + 1))                         # [Line 91]
                update_state                                        # [Line 92]
            done

            # Finished processing current user's repos; clear the current user state to move to next user
            CURRENT_USER=""                                         # [Line 96]
            CURRENT_REPO_INDEX=0                                    # [Line 97]
            update_state                                            # [Line 98]
        done

        # After processing all users in the batch, update SINCE to the last user ID
        SINCE=$LAST_ID                                            # [Line 102]
        update_state                                            # [Line 103]
    done
}

# 5. Start the process
fetch_users                                                    # [Line 107]

