# Accessing Remote SQLite Database via SSHFS

This guide explains how to set up SSHFS (SSH Filesystem) for accessing a SQLite database located on a remote server.
SSHFS allows you to securely mount a remote file system over SSH, making the remote database accessible as if it were on
your local file system.

## Prerequisites

- SSH access to the remote server where the SQLite database is located.
- SSHFS installed on your local machine.
- A user account with sufficient privileges to install packages and mount file systems on your local machine.

## Step 1: Install SSHFS

### For Linux:

```bash
sudo apt-get install sshfs
```

### For macOS:

Use [FUSE for macOS](https://osxfuse.github.io/).

## Step 2: Mount the Remote Directory

Create a local directory to serve as the mount point for the remote file system, and then mount the remote directory.

```bash
# Create a local directory for the mount point
mkdir ~/remote_db

# Mount the remote directory
sshfs <username>@<remote_ip>:/mnt/sdb1/DB/ ~/remote_db
```

Replace `<username>` with your SSH username and `<remote_ip>` with the IP address of the remote server.

## Step 3: Access the Database

After mounting, the SQLite database file located in `/mnt/sdb1/DB/` on the remote server will be accessible
at `~/remote_db/` on your local machine.

## Step 4: Unmount the Remote Directory

When you are done, you can unmount the remote directory:

### For Linux:

```bash
fusermount -u ~/remote_db
```

### For macOS:

```bash
umount ~/remote_db
```

## Step 5: Update Your Flask Application (Optional)

If you are accessing this database in a Flask application, update the `DB_PATH` configuration in your Flask app to point
to the mounted path:

```python
app.config['DB_PATH'] = '/home/your_local_username/remote_db/your_database_file.db'
```

## Security Considerations

- Use SSH keys instead of passwords for more secure SSH connections.
- Ensure your firewall and network security settings allow for secure SSH access.
