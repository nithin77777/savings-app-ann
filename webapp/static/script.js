/**
 * ANN Savings Calculator - Frontend JavaScript
 *
 * This script handles the interactive functionality of the web application,
 * including form submission, API communication, result display, and chart
 * visualization using Chart.js library.
 *
 * Key Features:
 * - Asynchronous form submission with validation
 * - Real-time API communication with Flask backend
 * - Dynamic result display with formatted output
 * - Interactive chart visualization with Chart.js
 * - Error handling and user feedback
 *
 * Author: ANN Savings Calculator Team
 * Created: 2025
 */

// Global variable to store chart instance for proper cleanup
let chartRef = null;

/**
 * Main form submission handler
 *
 * This function is triggered when users submit the financial input form.
 * It handles the complete prediction workflow from data collection to
 * result visualization.
 */
document.getElementById("form").addEventListener("submit", async (e) => {
	// Prevent default form submission behavior
	e.preventDefault();

	try {
		// === DATA COLLECTION AND PREPROCESSING ===

		// Extract form data using FormData API
		const formData = new FormData(e.target);

		// Convert FormData to JavaScript object
		const payload = Object.fromEntries(formData.entries());

		// Convert string inputs to numbers for API consumption
		// Required fields: income, spending, savings_goal
		for (const key of ["income", "spending", "savings_goal"]) {
			payload[key] = Number(payload[key]);
		}

		// === API COMMUNICATION ===

		// Send prediction request to Flask backend
		const response = await fetch("/api/predict", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(payload),
		});

		// Parse JSON response from API
		const data = await response.json();

		// Handle API errors gracefully
		if (!data.ok) {
			alert(data.error || "Prediction API error occurred");
			return;
		}

		// === RESULT DISPLAY ===

		// Extract prediction results from API response
		const result = data.result;

		// Show the results section
		const resultSection = document.getElementById("result");
		resultSection.style.display = "block";

		// Update individual result elements with predictions
		document.getElementById("spend").textContent =
			result.spending_percentage + "%";
		document.getElementById("save").textContent =
			result.savings_percentage + "%";
		document.getElementById("goal").textContent = result.goal_achieved
			? "YES"
			: "NO";
		document.getElementById("rec").textContent = result.recommendation;
		document.getElementById("note").textContent = result.note;

		// === CHART VISUALIZATION ===

		// Get chart canvas element and display it
		const canvas = document.getElementById("chart");
		canvas.style.display = "block";
		const ctx = canvas.getContext("2d");

		// Destroy existing chart instance to prevent memory leaks
		if (chartRef) {
			chartRef.destroy();
		}

		// Create new interactive bar chart using Chart.js
		chartRef = new Chart(ctx, {
			type: "bar",
			data: {
				// Chart categories for comparison
				labels: ["Spending %", "Saving %", "Goal %"],
				datasets: [
					{
						label: "Financial Breakdown",
						data: [
							result.spending_percentage, // Predicted spending %
							result.savings_percentage, // Predicted savings %
							payload.savings_goal, // Target goal %
						],
						backgroundColor: [
							"rgba(255, 99, 132, 0.8)", // Red for spending
							"rgba(54, 162, 235, 0.8)", // Blue for savings
							"rgba(255, 205, 86, 0.8)", // Yellow for goal
						],
						borderColor: [
							"rgba(255, 99, 132, 1)",
							"rgba(54, 162, 235, 1)",
							"rgba(255, 205, 86, 1)",
						],
						borderWidth: 2,
					},
				],
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				plugins: {
					title: {
						display: true,
						text: "Financial Analysis Results",
					},
					legend: {
						display: false, // Hide legend for cleaner look
					},
				},
				scales: {
					y: {
						beginAtZero: true,
						max: 100,
						title: {
							display: true,
							text: "Percentage (%)",
						},
					},
					x: {
						title: {
							display: true,
							text: "Categories",
						},
					},
				},
			},
		});

		// Scroll to results for better user experience
		resultSection.scrollIntoView({ behavior: "smooth" });
	} catch (error) {
		// Handle network errors and unexpected issues
		console.error("Prediction error:", error);
		alert(
			"Unable to get prediction. Please check your connection and try again."
		);
	}
});
