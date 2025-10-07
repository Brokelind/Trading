# deploy_gh_pages.py
import os
import shutil
import subprocess
from web_dashboard import generate_dashboard

def deploy_to_gh_pages():
    print("🚀 Deploying to GitHub Pages...")
    
    # Generate fresh dashboard
    generate_dashboard()
    
    # Switch to gh-pages branch
    subprocess.run(["git", "checkout", "gh-pages"])
    
    # Copy web_dashboard contents to root
    if os.path.exists("web_dashboard"):
        # Copy all files from web_dashboard to root
        for item in os.listdir("web_dashboard"):
            src = os.path.join("web_dashboard", item)
            dst = item
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    
    # Commit and push
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Update trading dashboard"])
    subprocess.run(["git", "push", "origin", "gh-pages"])
    
    # Switch back to main branch
    subprocess.run(["git", "checkout", "main"])
    
    print("✅ Deployed! Your dashboard is live at:")
    print(f"   https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/")

if __name__ == "__main__":
    deploy_to_gh_pages()