# Connect Github to Server

Cloning of private repos does not work because the user is not authenticated.
An SSH key needs to be created to do the authentication.
1. To create an SSH key, use `ssh-keygen -t ed25519 -C "<email>"`. Press enter / accept when questions occure.
2. Check if SSH agent is running: `eval "$(ssh-agent -s)"`. A pid should be printed.
3. Add the SSH private key to the SSH agent: `ssh-add ~/.ssh/id_ed25519`
4. To print the SSH key itself, use: `cat ~/.ssh/id_ed25519.pub`
5. The printed SSH key needs to be added to the Github account logged in: Settings -> SSH and GPG keys -> New SSH key
6. Use `ssh -T git@github.com` to authenticate.
After authentication, `git clone` should work even for private repos.