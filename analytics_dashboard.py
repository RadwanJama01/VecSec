"""
Analytics Dashboard for VecSec Red Team Framework
=================================================

This module provides a comprehensive analytics dashboard for visualizing
and analyzing attack results, vulnerability discoveries, and system
performance metrics from the adversarial testing framework.

Author: VecSec Team
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from dataclasses import asdict

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.utils
import pandas as pd

from attack_logger import AttackLogger

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """
    Analytics dashboard for attack results and vulnerability analysis.
    
    Features:
    - Real-time attack statistics
    - Vulnerability tracking
    - Performance metrics
    - Interactive visualizations
    - Export capabilities
    - Alert system
    """
    
    def __init__(self, 
                 attack_logger: AttackLogger,
                 config: Optional[Dict] = None):
        self.attack_logger = attack_logger
        self.config = config or {}
        
        # Flask app setup
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Dashboard configuration
        self.refresh_interval = self.config.get('refresh_interval', 30)  # seconds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'critical_vulnerabilities': 1,
            'high_evasion_rate': 0.3,
            'low_detection_rate': 0.5
        })
        
        # Setup routes
        self._setup_routes()
        
        logger.info("AnalyticsDashboard initialized")
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/stats')
        async def get_stats():
            """Get current attack statistics."""
            try:
                stats = await self.attack_logger.get_attack_statistics()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vulnerabilities')
        async def get_vulnerabilities():
            """Get discovered vulnerabilities."""
            try:
                limit = request.args.get('limit', 10, type=int)
                vulnerabilities = await self.attack_logger.get_top_vulnerabilities(limit)
                return jsonify(vulnerabilities)
            except Exception as e:
                logger.error(f"Failed to get vulnerabilities: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/attacks/timeline')
        async def get_attack_timeline():
            """Get attack timeline data."""
            try:
                hours = request.args.get('hours', 24, type=int)
                timeline_data = await self._get_attack_timeline(hours)
                return jsonify(timeline_data)
            except Exception as e:
                logger.error(f"Failed to get attack timeline: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/attacks/by-mode')
        async def get_attacks_by_mode():
            """Get attack breakdown by mode."""
            try:
                mode_data = await self._get_attacks_by_mode()
                return jsonify(mode_data)
            except Exception as e:
                logger.error(f"Failed to get attacks by mode: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/attacks/by-method')
        async def get_attacks_by_method():
            """Get attack breakdown by method."""
            try:
                method_data = await self._get_attacks_by_method()
                return jsonify(method_data)
            except Exception as e:
                logger.error(f"Failed to get attacks by method: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance')
        async def get_performance_metrics():
            """Get performance metrics."""
            try:
                performance_data = await self._get_performance_metrics()
                return jsonify(performance_data)
            except Exception as e:
                logger.error(f"Failed to get performance metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        async def get_alerts():
            """Get current alerts."""
            try:
                alerts = await self._get_alerts()
                return jsonify(alerts)
            except Exception as e:
                logger.error(f"Failed to get alerts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export')
        async def export_data():
            """Export attack data."""
            try:
                start_time = request.args.get('start_time')
                end_time = request.args.get('end_time')
                
                start_dt = datetime.fromisoformat(start_time) if start_time else None
                end_dt = datetime.fromisoformat(end_time) if end_time else None
                
                output_path = f"./exports/attack_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                await self.attack_logger.export_attack_data(
                    output_path, start_dt, end_dt
                )
                
                return jsonify({'export_path': output_path})
            except Exception as e:
                logger.error(f"Failed to export data: {e}")
                return jsonify({'error': str(e)}), 500
    
    async def _get_attack_timeline(self, hours: int) -> Dict:
        """Get attack timeline data for visualization."""
        conn = sqlite3.connect(self.attack_logger.db_path)
        cursor = conn.cursor()
        
        try:
            # Get attacks by hour
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', executed_at) as hour,
                    COUNT(*) as total_attacks,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attacks,
                    SUM(CASE WHEN evasion_successful = 1 THEN 1 ELSE 0 END) as evasion_successes,
                    SUM(CASE WHEN detected_as_malicious = 1 THEN 1 ELSE 0 END) as detections
                FROM attack_executions
                WHERE executed_at >= datetime('now', '-{} hours')
                GROUP BY hour
                ORDER BY hour
            """.format(hours))
            
            timeline_data = cursor.fetchall()
            
            return {
                'timeline': [
                    {
                        'hour': row[0],
                        'total_attacks': row[1],
                        'successful_attacks': row[2],
                        'evasion_successes': row[3],
                        'detections': row[4]
                    }
                    for row in timeline_data
                ]
            }
            
        finally:
            conn.close()
    
    async def _get_attacks_by_mode(self) -> Dict:
        """Get attack breakdown by mode."""
        conn = sqlite3.connect(self.attack_logger.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    mode,
                    COUNT(*) as count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN evasion_successful = 1 THEN 1 ELSE 0 END) as evasions,
                    AVG(execution_duration_ms) as avg_duration
                FROM attack_executions
                WHERE executed_at >= datetime('now', '-24 hours')
                GROUP BY mode
            """)
            
            mode_data = cursor.fetchall()
            
            return {
                'modes': [
                    {
                        'mode': row[0],
                        'count': row[1],
                        'successes': row[2],
                        'evasions': row[3],
                        'avg_duration': row[4]
                    }
                    for row in mode_data
                ]
            }
            
        finally:
            conn.close()
    
    async def _get_attacks_by_method(self) -> Dict:
        """Get attack breakdown by method."""
        conn = sqlite3.connect(self.attack_logger.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    method,
                    COUNT(*) as count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN evasion_successful = 1 THEN 1 ELSE 0 END) as evasions
                FROM attack_executions
                WHERE executed_at >= datetime('now', '-24 hours')
                GROUP BY method
                ORDER BY successes DESC
                LIMIT 10
            """)
            
            method_data = cursor.fetchall()
            
            return {
                'methods': [
                    {
                        'method': row[0],
                        'count': row[1],
                        'successes': row[2],
                        'evasions': row[3]
                    }
                    for row in method_data
                ]
            }
            
        finally:
            conn.close()
    
    async def _get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        conn = sqlite3.connect(self.attack_logger.db_path)
        cursor = conn.cursor()
        
        try:
            # Get execution time statistics
            cursor.execute("""
                SELECT 
                    AVG(execution_duration_ms) as avg_duration,
                    MIN(execution_duration_ms) as min_duration,
                    MAX(execution_duration_ms) as max_duration,
                    COUNT(*) as total_attacks
                FROM attack_executions
                WHERE executed_at >= datetime('now', '-24 hours')
            """)
            
            perf_stats = cursor.fetchone()
            
            # Get sandbox statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT sandbox_id) as active_sandboxes,
                    AVG(execution_duration_ms) as avg_sandbox_time
                FROM attack_executions
                WHERE executed_at >= datetime('now', '-1 hour')
            """)
            
            sandbox_stats = cursor.fetchone()
            
            return {
                'execution_time': {
                    'avg_duration_ms': perf_stats[0] or 0,
                    'min_duration_ms': perf_stats[1] or 0,
                    'max_duration_ms': perf_stats[2] or 0,
                    'total_attacks': perf_stats[3] or 0
                },
                'sandbox_stats': {
                    'active_sandboxes': sandbox_stats[0] or 0,
                    'avg_sandbox_time_ms': sandbox_stats[1] or 0
                }
            }
            
        finally:
            conn.close()
    
    async def _get_alerts(self) -> List[Dict]:
        """Get current alerts based on thresholds."""
        alerts = []
        
        try:
            # Get current statistics
            stats = await self.attack_logger.get_attack_statistics()
            
            # Check for critical vulnerabilities
            vulnerabilities = await self.attack_logger.get_top_vulnerabilities(5)
            critical_vulns = [v for v in vulnerabilities if v['severity'] == 'critical']
            
            if len(critical_vulns) >= self.alert_thresholds['critical_vulnerabilities']:
                alerts.append({
                    'type': 'critical',
                    'message': f"{len(critical_vulns)} critical vulnerabilities discovered",
                    'timestamp': datetime.now().isoformat(),
                    'action_required': True
                })
            
            # Check evasion rate
            evasion_rate = stats['overall']['evasion_rate']
            if evasion_rate >= self.alert_thresholds['high_evasion_rate']:
                alerts.append({
                    'type': 'warning',
                    'message': f"High evasion rate detected: {evasion_rate:.2%}",
                    'timestamp': datetime.now().isoformat(),
                    'action_required': True
                })
            
            # Check detection rate
            detection_rate = stats['overall']['detection_rate']
            if detection_rate <= self.alert_thresholds['low_detection_rate']:
                alerts.append({
                    'type': 'warning',
                    'message': f"Low detection rate: {detection_rate:.2%}",
                    'timestamp': datetime.now().isoformat(),
                    'action_required': True
                })
            
        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
            alerts.append({
                'type': 'error',
                'message': f"Alert system error: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'action_required': False
            })
        
        return alerts
    
    def create_visualizations(self, data: Dict) -> Dict[str, str]:
        """Create Plotly visualizations for the dashboard."""
        visualizations = {}
        
        try:
            # Attack timeline chart
            if 'timeline' in data:
                timeline_fig = go.Figure()
                
                timeline_data = data['timeline']
                hours = [row['hour'] for row in timeline_data]
                
                timeline_fig.add_trace(go.Scatter(
                    x=hours,
                    y=[row['total_attacks'] for row in timeline_data],
                    mode='lines+markers',
                    name='Total Attacks',
                    line=dict(color='blue')
                ))
                
                timeline_fig.add_trace(go.Scatter(
                    x=hours,
                    y=[row['successful_attacks'] for row in timeline_data],
                    mode='lines+markers',
                    name='Successful Attacks',
                    line=dict(color='green')
                ))
                
                timeline_fig.add_trace(go.Scatter(
                    x=hours,
                    y=[row['evasion_successes'] for row in timeline_data],
                    mode='lines+markers',
                    name='Evasion Successes',
                    line=dict(color='red')
                ))
                
                timeline_fig.update_layout(
                    title='Attack Timeline',
                    xaxis_title='Time',
                    yaxis_title='Number of Attacks',
                    hovermode='x unified'
                )
                
                visualizations['timeline'] = json.dumps(
                    timeline_fig, cls=plotly.utils.PlotlyJSONEncoder
                )
            
            # Attack mode pie chart
            if 'modes' in data:
                mode_fig = go.Figure(data=[go.Pie(
                    labels=[mode['mode'] for mode in data['modes']],
                    values=[mode['count'] for mode in data['modes']],
                    hole=0.3
                )])
                
                mode_fig.update_layout(
                    title='Attacks by Mode',
                    showlegend=True
                )
                
                visualizations['modes'] = json.dumps(
                    mode_fig, cls=plotly.utils.PlotlyJSONEncoder
                )
            
            # Attack method bar chart
            if 'methods' in data:
                method_fig = go.Figure(data=[go.Bar(
                    x=[method['method'] for method in data['methods']],
                    y=[method['successes'] for method in data['methods']],
                    name='Successful Attacks'
                )])
                
                method_fig.add_trace(go.Bar(
                    x=[method['method'] for method in data['methods']],
                    y=[method['evasions'] for method in data['methods']],
                    name='Evasion Successes'
                ))
                
                method_fig.update_layout(
                    title='Attacks by Method',
                    xaxis_title='Attack Method',
                    yaxis_title='Number of Attacks',
                    barmode='group'
                )
                
                visualizations['methods'] = json.dumps(
                    method_fig, cls=plotly.utils.PlotlyJSONEncoder
                )
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
        
        return visualizations
    
    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """Run the analytics dashboard."""
        logger.info(f"Starting analytics dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# HTML Template for Dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VecSec Red Team Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .alerts-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert.critical {
            background-color: #ffebee;
            border-color: #f44336;
            color: #c62828;
        }
        .alert.warning {
            background-color: #fff3e0;
            border-color: #ff9800;
            color: #e65100;
        }
        .alert.info {
            background-color: #e3f2fd;
            border-color: #2196f3;
            color: #1565c0;
        }
        .refresh-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
        }
        .export-btn {
            background: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ VecSec Red Team Analytics Dashboard</h1>
        <p>Real-time attack monitoring and vulnerability analysis</p>
        <button class="refresh-btn" onclick="refreshDashboard()">Refresh</button>
        <button class="export-btn" onclick="exportData()">Export Data</button>
    </div>

    <div class="stats-grid" id="statsGrid">
        <!-- Stats will be populated by JavaScript -->
    </div>

    <div class="chart-container">
        <h3>Attack Timeline</h3>
        <div id="timelineChart"></div>
    </div>

    <div class="chart-container">
        <h3>Attacks by Mode</h3>
        <div id="modeChart"></div>
    </div>

    <div class="chart-container">
        <h3>Attacks by Method</h3>
        <div id="methodChart"></div>
    </div>

    <div class="alerts-container">
        <h3>Alerts</h3>
        <div id="alertsContainer">
            <!-- Alerts will be populated by JavaScript -->
        </div>
    </div>

    <script>
        let refreshInterval;

        async function refreshDashboard() {
            try {
                // Update stats
                const statsResponse = await fetch('/api/stats');
                const stats = await statsResponse.json();
                updateStatsGrid(stats);

                // Update charts
                const timelineResponse = await fetch('/api/attacks/timeline');
                const timelineData = await timelineResponse.json();
                updateTimelineChart(timelineData);

                const modeResponse = await fetch('/api/attacks/by-mode');
                const modeData = await modeResponse.json();
                updateModeChart(modeData);

                const methodResponse = await fetch('/api/attacks/by-method');
                const methodData = await methodResponse.json();
                updateMethodChart(methodData);

                // Update alerts
                const alertsResponse = await fetch('/api/alerts');
                const alerts = await alertsResponse.json();
                updateAlerts(alerts);

            } catch (error) {
                console.error('Failed to refresh dashboard:', error);
            }
        }

        function updateStatsGrid(stats) {
            const statsGrid = document.getElementById('statsGrid');
            const overall = stats.overall;

            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${overall.total_attacks}</div>
                    <div class="stat-label">Total Attacks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(overall.success_rate * 100).toFixed(1)}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(overall.evasion_rate * 100).toFixed(1)}%</div>
                    <div class="stat-label">Evasion Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(overall.detection_rate * 100).toFixed(1)}%</div>
                    <div class="stat-label">Detection Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${overall.campaigns_count}</div>
                    <div class="stat-label">Campaigns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.vulnerability_stats.reduce((sum, v) => sum + v.count, 0)}</div>
                    <div class="stat-label">Vulnerabilities</div>
                </div>
            `;
        }

        function updateTimelineChart(data) {
            const timeline = data.timeline;
            
            const trace1 = {
                x: timeline.map(t => t.hour),
                y: timeline.map(t => t.total_attacks),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Total Attacks',
                line: { color: 'blue' }
            };

            const trace2 = {
                x: timeline.map(t => t.hour),
                y: timeline.map(t => t.successful_attacks),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Successful Attacks',
                line: { color: 'green' }
            };

            const trace3 = {
                x: timeline.map(t => t.hour),
                y: timeline.map(t => t.evasion_successes),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Evasion Successes',
                line: { color: 'red' }
            };

            const layout = {
                title: 'Attack Timeline',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Number of Attacks' },
                hovermode: 'x unified'
            };

            Plotly.newPlot('timelineChart', [trace1, trace2, trace3], layout);
        }

        function updateModeChart(data) {
            const modes = data.modes;
            
            const trace = {
                labels: modes.map(m => m.mode),
                values: modes.map(m => m.count),
                type: 'pie',
                hole: 0.3
            };

            const layout = {
                title: 'Attacks by Mode'
            };

            Plotly.newPlot('modeChart', [trace], layout);
        }

        function updateMethodChart(data) {
            const methods = data.methods;
            
            const trace1 = {
                x: methods.map(m => m.method),
                y: methods.map(m => m.successes),
                type: 'bar',
                name: 'Successful Attacks'
            };

            const trace2 = {
                x: methods.map(m => m.method),
                y: methods.map(m => m.evasions),
                type: 'bar',
                name: 'Evasion Successes'
            };

            const layout = {
                title: 'Attacks by Method',
                xaxis: { title: 'Attack Method' },
                yaxis: { title: 'Number of Attacks' },
                barmode: 'group'
            };

            Plotly.newPlot('methodChart', [trace1, trace2], layout);
        }

        function updateAlerts(alerts) {
            const alertsContainer = document.getElementById('alertsContainer');
            
            if (alerts.length === 0) {
                alertsContainer.innerHTML = '<p>No alerts at this time.</p>';
                return;
            }

            alertsContainer.innerHTML = alerts.map(alert => `
                <div class="alert ${alert.type}">
                    <strong>${alert.type.toUpperCase()}:</strong> ${alert.message}
                    <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }

        async function exportData() {
            try {
                const response = await fetch('/api/export');
                const result = await response.json();
                
                if (result.export_path) {
                    alert(`Data exported to: ${result.export_path}`);
                } else {
                    alert('Export failed');
                }
            } catch (error) {
                console.error('Export failed:', error);
                alert('Export failed');
            }
        }

        // Auto-refresh every 30 seconds
        function startAutoRefresh() {
            refreshInterval = setInterval(refreshDashboard, 30000);
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            refreshDashboard();
            startAutoRefresh();
        });
    </script>
</body>
</html>
"""

# Example usage
if __name__ == "__main__":
    # This would be used with actual implementations
    pass
