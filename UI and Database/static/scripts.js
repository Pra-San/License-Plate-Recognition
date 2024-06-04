document.addEventListener('DOMContentLoaded', function() {
    const entryForm = document.getElementById('entry-form');
    const exitForm = document.getElementById('exit-form');
    const fetchEntriesForm = document.getElementById('fetch-entries-form');
    const refreshEntriesButton = document.getElementById('refresh-entries');
    const entriesTableBody = document.querySelector('#entries-table tbody');
    const vehicleEntriesList = document.getElementById('vehicle-entries-list');

    entryForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const licensePlate = document.getElementById('entry-license-plate').value;
        fetch(`/vehicle_entry?license_plate=${licensePlate}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => alert(data.message))
            .catch(error => console.error('Error:', error));
    });

    exitForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const licensePlate = document.getElementById('exit-license-plate').value;
        fetch(`/vehicle_exit?license_plate=${licensePlate}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => alert(data.message))
            .catch(error => console.error('Error:', error));
    });

    fetchEntriesForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const licensePlate = document.getElementById('fetch-license-plate').value;
        fetch(`/vehicle_entries/?license_plate=${licensePlate}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('No Entries');
                }
                return response.json();
            })
            .then(entries => {
                vehicleEntriesList.innerHTML = '';
                if (entries.length === 0) {
                    const li = document.createElement('li');
                    li.textContent = 'No Entries';
                    vehicleEntriesList.appendChild(li);
                } else {
                    entries.forEach(entry => {
                        const li = document.createElement('li');
                        li.textContent = `Entry Time: ${entry.entry_time}, Exit Time: ${entry.exit_time || "Still parked"}`;
                        vehicleEntriesList.appendChild(li);
                    });
                }
            })
            .catch(error => {
                vehicleEntriesList.innerHTML = '';
                const li = document.createElement('li');
                li.textContent = error.message;
                vehicleEntriesList.appendChild(li);
            });
    });
    refreshEntriesButton.addEventListener('click', function() {
        fetch('/recent_entries')
            .then(response => response.json())
            .then(entries => {
                entriesTableBody.innerHTML = '';
                entries.forEach(entry => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${entry.license_plate}</td>
                        <td>${entry.entry_time}</td>
                        <td>${entry.exit_time || "Still parked"}</td>
                    `;
                    entriesTableBody.appendChild(row);
                });
            })
            .catch(error => console.error('Error:', error));
    });

    // Fetch recent entries on page load
    refreshEntriesButton.click();
});
