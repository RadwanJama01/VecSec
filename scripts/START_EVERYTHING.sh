#!/bin/bash
# VecSec Complete Startup Script
# Starts Docker monitoring, then runs the security tests

echo "🚀 VecSec Complete Startup"
echo "=========================="
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "   Please start Docker Desktop first"
    exit 1
fi

# Start monitoring services
echo "📊 Starting Prometheus & Grafana..."
docker-compose -f docker-compose.monitoring.yml up -d
sleep 3

# Check if containers are running
if docker ps | grep -q vecsec_prometheus && docker ps | grep -q vecsec_grafana; then
    echo "✅ Monitoring services started!"
else
    echo "⚠️  Monitoring services may not be fully ready yet"
fi

echo ""
echo "========================================="
echo "🎯 VecSec is now running!"
echo "========================================="
echo ""
echo "📊 Access Dashboard:"
echo "   - Grafana:    http://localhost:3000"
echo "   - Prometheus: http://localhost:9090"
echo ""
echo "🔑 Grafana Login:"
echo "   Username: admin"
echo "   Password: vecsec_admin"
echo ""
echo "🧪 Run Tests:"
echo "   python3 Good_Vs_Evil.py --test-type blind --blind-tests 20"
echo ""
echo "📈 View Metrics (Simple):"
echo "   python3 SIMPLE_METRICS_VIEWER.py"
echo ""
echo "💡 Or watch metrics live:"
echo "   watch -n 2 python3 SIMPLE_METRICS_VIEWER.py"
echo ""
echo "🛑 Stop Services:"
echo "   docker-compose -f docker-compose.monitoring.yml down"
echo ""

