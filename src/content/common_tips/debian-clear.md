---
title: "Debian virtual machine cache cleanup"
description: "Tips for cleaning up the cache in your Debian virtual machine"
pubDate: 2026-02-14
tags: ["Debian", "cleanup"]
heroImage: "/images/debian.png"
---

Sometimes, our virtual machines, such as Ubuntu and Debian,are installed on the E drive. Simply cleaning up the cache won't remove it. Here's a great method to successfully clean up and release the cache.

# I. First, check which directory is taking up space

Execute:

```bash
df -h
```
Check which directory is full: `/`, `/home`, or `/mnt/e`.

Then check the main directories:

```bash
sudo du -xh / --max-depth=1 | sort -h
```
The directories that usually take up space are:

* `/var`

* `/home`

* `/usr`

* `/opt`

* `/tmp`

# II. APT cache cleanup (usually releases 1~5GB)

```bash
sudo apt clean
sudo apt autoclean
sudo apt autoremove -y
```

Notes:

* `apt clean`: Deletes all downloaded deb package caches
* `apt autoclean`: Deletes expired packages
* `apt autoremove`: Deletes unused dependencies

Cache location:

```bash
/var/cache/apt/archives/
```

# III. Clean up journal logs

Check size:

```bash
sudo du -sh /var/log/journal
```

Limit to 200MB:

```bash
sudo journalctl --vacuum-size=200M
```

Or only keep 7 days:

```bash
sudo journalctl --vacuum-time=7d
```

# IV. Clean up /var/log old logs

```bash
sudo du -sh /var/log/*
```

If you see some huge files:

```bash
sudo truncate -s 0 /var/log/syslog
sudo truncate -s 0 /var/log/kern.log
sudo truncate -s 0 /var/log/auth.log
```

Or:

```bash
sudo rm -f /var/log/*.gz
sudo rm -f /var/log/*.[0-9]
```

# V. pip / conda / python cache

### pip cache:

```bash
pip cache purge
```

Or manually:

```bash
rm -rf ~/.cache/pip
```

### conda (if installed):

```bash
conda clean --all -y
```

# VI. CUDA / Compilation cache

### NVCC / torch cache:

```bash
rm -rf ~/.nv
rm -rf ~/.cache/torch
rm -rf ~/.cache/nv
rm -rf ~/.cache
```

### CMake / build intermediate files (very large):

For example, you have:

```
~/GOSK/build
~/xxx/build
```

Directly:

```bash
rm -rf build/
```

Or:

```bash
find ~ -name build -type d -exec rm -rf {} +
```

# VII. Clean up Docker (if installed)

```bash
docker system df
docker system prune -a
```

This will delete all unused images


# VIII. Find 10GB+ files (recommended)

```bash
sudo find / -type f -size +10G 2>/dev/null
```

Or:

```bash
sudo du -ah / | sort -rh | head -n 50
```

# IX. If you use WSL

If your Debian is **WSL and placed on E disk**:

Even if you delete the file, **the virtual disk file .vhdx will not automatically shrink!**

You need to do:

### Turn off WSL

In Windows PowerShell (administrator):

```powershell
wsl --shutdown
```

### Compress virtual disk

Open:

> Disk Management → Operation → Attach VHD
> Find your Debian's `.vhdx`

Then:

> Right-click → Compress Volume

Or use:

```powershell
diskpart
select vdisk file="E:\xxx\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

Otherwise, your **disk will still look full**
