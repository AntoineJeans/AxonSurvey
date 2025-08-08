let groupCount = 1;
let rats = [];
let regions = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get rats and regions from the page data
    rats = JSON.parse(document.getElementById('rats-data').textContent);
    regions = JSON.parse(document.getElementById('regions-data').textContent);
    
    // Initialize the first tab with click handler
    const firstTab = document.querySelector('.group-tab[data-group="1"]');
    if (firstTab) {
        firstTab.onclick = () => switchGroup(1);
    }
});

function createNewGroup() {
    groupCount++;
    const currentGroupNum = groupCount; // Capture the current group number
    
    // Create new tab
    const newTab = document.createElement('div');
    newTab.className = 'group-tab';
    newTab.setAttribute('data-group', currentGroupNum);
    newTab.textContent = `Group ${currentGroupNum}`;
    newTab.onclick = () => switchGroup(currentGroupNum);
    document.getElementById('groupTabs').appendChild(newTab);
    
    // Create new content
    const newContent = document.createElement('div');
    newContent.className = 'group-content';
    newContent.setAttribute('data-group', currentGroupNum);
    
    // Build complete HTML structure
    let contentHtml = '';
    
    // Rats section
    contentHtml += '<div class="selector-section">';
    contentHtml += '<h3>🐭 Rats Selector</h3>';
    contentHtml += '<div class="checkbox-list">';
    contentHtml += `<div class="checkbox-item">`;
    contentHtml += `<input type="checkbox" id="all-rats-${currentGroupNum}" name="all-rats-${currentGroupNum}" onchange="toggleAllRats(${currentGroupNum})">`;
    contentHtml += `<label for="all-rats-${currentGroupNum}"><strong>All rats</strong></label>`;
    contentHtml += '</div>';
    
    rats.forEach(rat => {
        contentHtml += '<div class="checkbox-item">';
        contentHtml += `<input type="checkbox" id="rat-${rat}-${currentGroupNum}" name="rats-${currentGroupNum}" value="${rat}" onchange="updateAllRatsCheckbox(${currentGroupNum})">`;
        contentHtml += `<label for="rat-${rat}-${currentGroupNum}">${rat}</label>`;
        contentHtml += '</div>';
    });
    contentHtml += '</div></div>';
    
    // Regions section
    contentHtml += '<div class="selector-section">';
    contentHtml += '<h3>🧠 Regions Selector</h3>';
    contentHtml += '<div class="checkbox-list">';
    contentHtml += `<div class="checkbox-item">`;
    contentHtml += `<input type="checkbox" id="all-regions-${currentGroupNum}" name="all-regions-${currentGroupNum}" onchange="toggleAllRegions(${currentGroupNum})">`;
    contentHtml += `<label for="all-regions-${currentGroupNum}"><strong>All regions</strong></label>`;
    contentHtml += '</div>';
    
    regions.forEach(region => {
        contentHtml += '<div class="checkbox-item">';
        contentHtml += `<input type="checkbox" id="region-${region}-${currentGroupNum}" name="regions-${currentGroupNum}" value="${region}" onchange="updateAllRegionsCheckbox(${currentGroupNum})">`;
        contentHtml += `<label for="region-${region}-${currentGroupNum}">${region}</label>`;
        contentHtml += '</div>';
    });
    contentHtml += '</div></div>';
    
    // Actions
    contentHtml += `<div class="group-actions">`;
    contentHtml += `<button type="button" class="btn btn-danger" onclick="deleteGroup(${currentGroupNum})">Delete Group</button>`;
    contentHtml += '</div>';
    
    newContent.innerHTML = contentHtml;
    document.getElementById('groupContents').appendChild(newContent);
    
    // Switch to new group
    switchGroup(currentGroupNum);
}

function switchGroup(groupNum) {
    // Hide all content
    document.querySelectorAll('.group-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active from all tabs
    document.querySelectorAll('.group-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected content and tab
    document.querySelector(`.group-content[data-group="${groupNum}"]`).classList.add('active');
    document.querySelector(`.group-tab[data-group="${groupNum}"]`).classList.add('active');
}

function deleteGroup(groupNum) {
    if (groupCount <= 1) {
        alert('Cannot delete the last group. At least one group is required.');
        return;
    }
    
    // Remove tab and content
    document.querySelector(`.group-tab[data-group="${groupNum}"]`).remove();
    document.querySelector(`.group-content[data-group="${groupNum}"]`).remove();
    
    groupCount--;
    
    // Renumber remaining groups to fix gaps
    renumberGroups();
    
    // Switch to first group if we deleted the active one
    if (document.querySelectorAll('.group-tab').length > 0) {
        const firstGroup = document.querySelector('.group-tab').getAttribute('data-group');
        switchGroup(firstGroup);
    }
}

function renumberGroups() {
    const tabs = document.querySelectorAll('.group-tab');
    const contents = document.querySelectorAll('.group-content');
    
    // Renumber tabs
    tabs.forEach((tab, index) => {
        const newGroupNum = index + 1;
        const oldGroupNum = tab.getAttribute('data-group');
        tab.setAttribute('data-group', newGroupNum);
        tab.textContent = `Group ${newGroupNum}`;
        
        // Update onclick handler
        tab.onclick = () => switchGroup(newGroupNum);
    });
    
    // Renumber contents and update all IDs and names
    contents.forEach((content, index) => {
        const newGroupNum = index + 1;
        const oldGroupNum = content.getAttribute('data-group');
        content.setAttribute('data-group', newGroupNum);
        
        // Update all input IDs and names
        const inputs = content.querySelectorAll('input');
        inputs.forEach(input => {
            if (input.id) {
                input.id = input.id.replace(`-${oldGroupNum}`, `-${newGroupNum}`);
            }
            if (input.name) {
                input.name = input.name.replace(`-${oldGroupNum}`, `-${newGroupNum}`);
            }
            if (input.onchange) {
                const onchangeStr = input.getAttribute('onchange');
                if (onchangeStr) {
                    input.setAttribute('onchange', onchangeStr.replace(`(${oldGroupNum})`, `(${newGroupNum})`));
                }
            }
        });
        
        // Update all label 'for' attributes
        const labels = content.querySelectorAll('label');
        labels.forEach(label => {
            if (label.getAttribute('for')) {
                label.setAttribute('for', label.getAttribute('for').replace(`-${oldGroupNum}`, `-${newGroupNum}`));
            }
        });
        
        // Update delete button onclick
        const deleteBtn = content.querySelector('.btn-danger');
        if (deleteBtn) {
            deleteBtn.onclick = () => deleteGroup(newGroupNum);
        }
    });
}

function toggleAllRats(groupNum) {
    const allRatsCheckbox = document.getElementById(`all-rats-${groupNum}`);
    const ratCheckboxes = document.querySelectorAll(`input[name="rats-${groupNum}"]`);
    
    ratCheckboxes.forEach(checkbox => {
        checkbox.checked = allRatsCheckbox.checked;
    });
}

function updateAllRatsCheckbox(groupNum) {
    const allRatsCheckbox = document.getElementById(`all-rats-${groupNum}`);
    const ratCheckboxes = document.querySelectorAll(`input[name="rats-${groupNum}"]`);
    const checkedRats = document.querySelectorAll(`input[name="rats-${groupNum}"]:checked`);
    
    allRatsCheckbox.checked = checkedRats.length === ratCheckboxes.length;
    allRatsCheckbox.indeterminate = checkedRats.length > 0 && checkedRats.length < ratCheckboxes.length;
}

function toggleAllRegions(groupNum) {
    const allRegionsCheckbox = document.getElementById(`all-regions-${groupNum}`);
    const regionCheckboxes = document.querySelectorAll(`input[name="regions-${groupNum}"]`);
    
    regionCheckboxes.forEach(checkbox => {
        checkbox.checked = allRegionsCheckbox.checked;
    });
}

function updateAllRegionsCheckbox(groupNum) {
    const allRegionsCheckbox = document.getElementById(`all-regions-${groupNum}`);
    const regionCheckboxes = document.querySelectorAll(`input[name="regions-${groupNum}"]`);
    const checkedRegions = document.querySelectorAll(`input[name="regions-${groupNum}"]:checked`);
    
    allRegionsCheckbox.checked = checkedRegions.length === regionCheckboxes.length;
    allRegionsCheckbox.indeterminate = checkedRegions.length > 0 && checkedRegions.length < regionCheckboxes.length;
}

function generateComparison() {
    // Collect all group data
    const groups = [];
    
    // Get all existing groups by their data-group attributes
    const existingGroups = document.querySelectorAll('.group-content');
    existingGroups.forEach((content, index) => {
        const groupNum = parseInt(content.getAttribute('data-group'));
        const ratCheckboxes = document.querySelectorAll(`input[name="rats-${groupNum}"]:checked`);
        const regionCheckboxes = document.querySelectorAll(`input[name="regions-${groupNum}"]:checked`);
        
        if (ratCheckboxes.length > 0 && regionCheckboxes.length > 0) {
            let selectedRats = Array.from(ratCheckboxes).map(cb => cb.value);
            let selectedRegions = Array.from(regionCheckboxes).map(cb => cb.value);
            
            // Check if "All rats" is selected
            const allRatsCheckbox = document.getElementById(`all-rats-${groupNum}`);
            if (allRatsCheckbox && allRatsCheckbox.checked) {
                selectedRats = ['ALL_RATS'];
            }
            
            // Check if "All regions" is selected
            const allRegionsCheckbox = document.getElementById(`all-regions-${groupNum}`);
            if (allRegionsCheckbox && allRegionsCheckbox.checked) {
                selectedRegions = ['ALL_REGIONS'];
            }
            
            groups.push({
                groupNum: groupNum,
                rats: selectedRats,
                regions: selectedRegions
            });
        }
    });
    
    if (groups.length < 2) {
        alert('Please define at least 2 groups with rats and regions selected.');
        return;
    }
    
    // Show loading state
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = '<div style="text-align: center; padding: 40px;"><strong>Generating comparison...</strong><br><small>This is a placeholder - actual comparison logic needs to be implemented</small></div>';
    
    // Call the API endpoint
    fetch('/api/compare/groups', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ groups: groups })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            outputArea.innerHTML = `<div style="color: red; text-align: center; padding: 40px;"><strong>Error:</strong> ${data.error}</div>`;
        } else {
            displayComparisonResults(groups, data);
        }
    })
    .catch(error => {
        outputArea.innerHTML = `<div style="color: red; text-align: center; padding: 40px;"><strong>Error:</strong> ${error.message}</div>`;
    });
}

function displayComparisonResults(groups, apiData) {
    const outputArea = document.getElementById('outputArea');
    
    let html = '<h3>📊 Comparison Results</h3>';
    html += '<div style="margin-bottom: 20px;">';
    html += '<strong>Groups defined:</strong><br>';
    groups.forEach(group => {
        html += `Group ${group.groupNum}: ${group.rats.length} rats, ${group.regions.length} regions<br>`;
    });
    html += '</div>';
    
    if (apiData && apiData.status === 'success') {
        html += '<div style="background: #d4edda; padding: 20px; border-radius: 5px; margin: 20px 0; border: 1px solid #c3e6cb;">';
        html += '<strong>✅ Analysis Completed:</strong><br>';
        html += `• Groups analyzed: ${apiData.groups_analyzed}<br>`;
        html += `• Total rats: ${apiData.summary.total_rats}<br>`;
        html += `• Total regions: ${apiData.summary.total_regions}<br>`;
        html += '</div>';
        
        html += '<div style="background: #e9ecef; padding: 20px; border-radius: 5px; margin: 20px 0;">';
        html += '<strong>📈 Results Summary:</strong><br>';
        html += `• Statistics: ${apiData.results.statistics}<br>`;
        html += `• Visualizations: ${apiData.results.visualizations}<br>`;
        html += `• Tests: ${apiData.results.tests}<br>`;
        html += '<br><em>Backend comparison logic needs to be implemented in app.py</em>';
        html += '</div>';
    } else {
        html += '<div style="background: #f8d7da; padding: 20px; border-radius: 5px; margin: 20px 0; border: 1px solid #f5c6cb;">';
        html += '<strong>❌ Analysis Failed:</strong><br>';
        html += apiData.message || 'Unknown error occurred';
        html += '</div>';
    }
    
    outputArea.innerHTML = html;
}
