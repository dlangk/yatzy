export function createChartConfig(labels, reverseCumulativeBins, currentBins, currentMaskBins) {
    return {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    type: "line",
                    label: "10k Optimal",
                    data: reverseCumulativeBins,
                    backgroundColor: "rgba(40, 150, 250, 0.1)",
                    borderColor: "rgba(40, 150, 250, 1)",
                    borderWidth: 1,
                    tension: 0.4,
                    yAxisID: "y1",
                },
                {
                    label: "Expected Scores Possible",
                    data: currentBins,
                    backgroundColor: "rgba(255, 100, 40, 0.9)",
                    borderColor: "rgba(255, 100, 40, 1)",
                    borderWidth: 1,
                    yAxisID: "y",
                },
                {
                    label: "Current Reroll Mask",
                    data: currentMaskBins,
                    backgroundColor: "rgba(50, 205, 50, 0.9)",
                    borderColor: "rgba(50, 205, 50, 1)",
                    borderWidth: 1,
                    yAxisID: "y",
                },
            ],
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: {title: {display: true, text: "Expected Scores"}},
                y: {type: "logarithmic", min: 1, max: 1000, position: "left"},
                y1: {min: 0, max: 1, position: "right"},
            },
        },
    };
}
