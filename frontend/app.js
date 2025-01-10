// Chart.js Configuration
Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Inter', sans-serif";

// Initialize Charts
function initializeCharts() {
    // Performance Trends Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    new Chart(performanceCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Active Policies',
                data: [65, 78, 82, 75, 85, 90],
                borderColor: '#2563eb',
                backgroundColor: '#2563eb20',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#e2e8f0'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    // Policy Distribution Chart
    const distributionCtx = document.getElementById('distributionChart').getContext('2d');
    new Chart(distributionCtx, {
        type: 'doughnut',
        data: {
            labels: ['Active', 'Pending', 'Expired', 'Cancelled'],
            datasets: [{
                data: [65, 15, 12, 8],
                backgroundColor: [
                    '#2563eb',
                    '#f59e0b',
                    '#64748b',
                    '#ef4444'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

// Navigation
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.add('hidden');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.remove('hidden');
    
    // Update active nav link
    document.querySelectorAll('nav a').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('onclick').includes(sectionId)) {
            link.classList.add('active');
        }
    });

    // Initialize charts if showing dashboard
    if (sectionId === 'dashboard') {
        initializeCharts();
    }
}

// File Upload Handling
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.querySelector('.upload-progress');

// Drag and drop handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary-color)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--border-color)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

async function handleFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    // Show progress
    dropZone.classList.add('hidden');
    uploadProgress.classList.remove('hidden');
    
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const result = await response.json();
        
        // Update UI with processing results
        updateDashboardMetrics(result);
        showSection('dashboard');
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload file. Please try again.');
        
        // Reset upload UI
        dropZone.classList.remove('hidden');
        uploadProgress.classList.add('hidden');
    }
}

// Update Dashboard Metrics
function updateDashboardMetrics(data) {
    // Update metric cards
    document.querySelector('.metric-value').textContent = data.totalAgents;
    document.querySelectorAll('.metric-value')[1].textContent = data.activePolicies;
    document.querySelectorAll('.metric-value')[2].textContent = data.retentionRate + '%';
    document.querySelectorAll('.metric-value')[3].textContent = data.avgPolicyDuration;

    // Update agent table
    const tbody = document.querySelector('.data-table tbody');
    tbody.innerHTML = ''; // Clear existing rows
    
    data.agents.forEach(agent => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${agent.name}</td>
            <td>${agent.activePolicies}</td>
            <td>${agent.retentionRate}%</td>
            <td>
                <div class="progress-bar">
                    <div class="progress" style="width: ${agent.performanceScore}%"></div>
                    <span>${agent.performanceScore}</span>
                </div>
            </td>
            <td>
                <button class="btn-view" onclick="viewReport('${agent.id}')">
                    <i class="fas fa-eye"></i> View
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });

    // Reinitialize charts with new data
    initializeCharts();
}

// View Report
async function viewReport(agentId) {
    try {
        const response = await fetch(`/api/reports/${agentId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch report');
        }
        
        const reportUrl = await response.text();
        window.open(reportUrl, '_blank');
        
    } catch (error) {
        console.error('Error viewing report:', error);
        alert('Failed to load report. Please try again.');
    }
}

// Initialize dashboard on load
document.addEventListener('DOMContentLoaded', () => {
    showSection('dashboard');
});

// Mobile Navigation Toggle
let isSidebarOpen = false;

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    isSidebarOpen = !isSidebarOpen;
    sidebar.classList.toggle('open', isSidebarOpen);
}

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
    const sidebar = document.querySelector('.sidebar');
    const toggleButton = document.querySelector('.toggle-sidebar');
    
    if (isSidebarOpen && 
        !sidebar.contains(e.target) && 
        !toggleButton.contains(e.target)) {
        toggleSidebar();
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.innerWidth > 768 && isSidebarOpen) {
        toggleSidebar();
    }
});
