#!/bin/bash
set -e

echo "=========================================="
echo "Creating .env.example Template Files"
echo "=========================================="
echo ""

# Create root .env.example
echo "📝 Creating .env.example at project root..."
cat > .env.example << 'EOF'
# WorldSystem Environment Configuration
# Copy this file to .env and fill in your values

# ==========================================
# User Configuration
# ==========================================
USERNAME=your_username
HOME=/home/your_username
WORKSPACE=/path/to/WorldSystem
UID=1000
GID=1000

# ==========================================
# Display Configuration (for GUI applications)
# ==========================================
X11SOCKET=/tmp/.X11-unix
XAUTHORITY=/home/your_username/.Xauthority
CUDA_PATH=/usr/local/cuda
LIBGL_ALWAYS_INDIRECT=1
DISPLAY=localhost:10.0
BIND_HOST=0.0.0.0

# ==========================================
# Data Configuration
# ==========================================
# Path to your input data/video segments
USE_FOLDER=/path/to/your/data

# ==========================================
# API Keys (Optional - only needed for specific features)
# ==========================================
# SerpAPI - For Google Lens object identification in frame_processor
# Get your key at: https://serpapi.com/manage-api-key
SERPAPI_KEY=your_serpapi_key_here

# OpenAI - Currently used for experimental features
# Get your key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_key_here

# Perplexity AI - For dimension estimation in frame_processor
# Get your key at: https://www.perplexity.ai/settings/api
PERPLEXITY_KEY=your_perplexity_key_here

# ==========================================
# Google Cloud Storage (Optional)
# ==========================================
# Path to your GCS service account credentials JSON file
# Create at: https://console.cloud.google.com/iam-admin/serviceaccounts
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
GCS_BUCKET_NAME=your-bucket-name

# ==========================================
# RabbitMQ Configuration (Optional - change for production)
# ==========================================
RABBITMQ_USER=admin
RABBITMQ_PASS=changeme_in_production

# ==========================================
# Monitoring (Optional - change for production)
# ==========================================
GRAFANA_USER=admin
GRAFANA_PASSWORD=changeme_in_production
EOF

echo "  ✓ Created: .env.example"

# Create docker/.env.example
echo ""
echo "📝 Creating docker/.env.example..."
cat > docker/.env.example << 'EOF'
# Docker-specific Environment Variables
# Copy this file to .env and fill in your values

# User Configuration
USERNAME=your_username
HOME=/home/your_username
WORKSPACE=/path/to/WorldSystem/test_container
UID=1000
GID=1000

# Display Configuration
X11SOCKET=/tmp/.X11-unix
XAUTHORITY=/home/your_username/.Xauthority
CUDA_PATH=/usr/local/cuda
LIBGL_ALWAYS_INDIRECT=1
DISPLAY=localhost:10.0
EOF

echo "  ✓ Created: docker/.env.example"

# Update .gitignore
echo ""
echo "📝 Updating .gitignore..."

# Check if .gitignore exists
if [ ! -f .gitignore ]; then
    echo "  ⚠️  .gitignore not found, creating new one..."
    touch .gitignore
fi

# Add environment file patterns if not already present
if ! grep -q "^# Environment files" .gitignore; then
    cat >> .gitignore << 'EOF'

# Environment files
.env
.env.*
!.env.example
docker/.env
!docker/.env.example

# Credentials and secrets
*credentials*.json
worldsystem-*.json
**/credentials/
*.pem
*.key
id_rsa*
*.p12

# API keys and tokens
.secrets
secrets/

EOF
    echo "  ✓ Updated .gitignore with environment file patterns"
else
    echo "  ○ .gitignore already has environment file patterns"
fi

echo ""
echo "=========================================="
echo "✅ Template Files Created!"
echo "=========================================="
echo ""
echo "Created files:"
echo "  - .env.example (root)"
echo "  - docker/.env.example"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and fill in your values"
echo "2. Copy docker/.env.example to docker/.env and fill in your values"
echo "3. Add these files to git:"
echo "   git add .env.example docker/.env.example .gitignore"
echo "   git commit -m 'Add environment variable templates and update .gitignore'"
echo ""
