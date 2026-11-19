(() => {
  const palette = {
    ink: "#14282d",
    teal: "#276f68",
    amber: "#bc7728",
    red: "#a24840",
    line: "#b5c5c8",
    paper: "#edf2f3",
  };
  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: { filename: "portfolio-ledger-chart", scale: 2 },
  };
  const layoutBase = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Aptos, Trebuchet MS, sans-serif", color: palette.ink, size: 13 },
    margin: { l: 60, r: 24, t: 24, b: 48 },
    hovermode: "x unified",
    xaxis: { gridcolor: "rgba(20,40,45,.10)", zeroline: false },
    yaxis: { gridcolor: "rgba(20,40,45,.10)", zeroline: false },
    legend: { orientation: "h", y: 1.1, x: 0 },
  };

  async function getJson(url) {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("Series data could not be loaded.");
    return response.json();
  }

  function money(value) {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(value);
  }

  function percent(value) {
    return value === null || value === undefined
      ? "—"
      : new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(value);
  }

  function monthEndRows(rows) {
    const byMonth = new Map();
    for (const row of rows) byMonth.set(row.date.slice(0, 7), row);
    return [...byMonth.values()];
  }

  function fillTable(body, rows, columns) {
    if (!body) return;
    body.replaceChildren();
    for (const row of monthEndRows(rows)) {
      const tr = document.createElement("tr");
      for (const column of columns) {
        const cell = document.createElement(column.header ? "th" : "td");
        if (column.header) cell.scope = "row";
        cell.textContent = column.format(row[column.key], row);
        tr.append(cell);
      }
      body.append(tr);
    }
  }

  async function renderDetail() {
    const ledger = document.getElementById("chart-ledger");
    if (!ledger || !window.Plotly) return;
    const runId = ledger.dataset.runId;
    const scope = ledger.dataset.scope;
    try {
      const rows = await getJson(`/api/runs/${encodeURIComponent(runId)}/series?scope=${encodeURIComponent(scope)}`);
      const dates = rows.map((row) => row.date);
      const firstEquity = rows[0]?.equity || 1;
      const firstBenchmark = rows.find((row) => Number.isFinite(row.benchmark))?.benchmark || 1;
      const traces = [
        {
          x: dates,
          y: rows.map((row) => (row.equity / firstEquity) * 100),
          type: "scatter",
          mode: "lines",
          name: "Portfolio",
          line: { color: palette.teal, width: 2.5 },
        },
      ];
      if (rows.some((row) => Number.isFinite(row.benchmark))) {
        traces.push({
          x: dates,
          y: rows.map((row) =>
            Number.isFinite(row.benchmark) ? (row.benchmark / firstBenchmark) * 100 : null
          ),
          type: "scatter",
          mode: "lines",
          name: "Benchmark",
          line: { color: palette.amber, width: 1.5, dash: "dot" },
        });
      }
      window.Plotly.newPlot("equity-chart", traces, {
        ...layoutBase,
        yaxis: { ...layoutBase.yaxis, title: "Indexed equity (start = 100)" },
      }, config);
      window.Plotly.newPlot("drawdown-chart", [
        {
          x: dates,
          y: rows.map((row) => row.drawdown),
          type: "scatter",
          mode: "lines",
          fill: "tozeroy",
          name: "Drawdown",
          line: { color: palette.red, width: 1.2 },
        },
      ], {
        ...layoutBase,
        yaxis: { ...layoutBase.yaxis, tickformat: ".0%", title: "Drawdown" },
      }, config);
      window.Plotly.newPlot("exposure-chart", [
        {
          x: dates,
          y: rows.map((row) => row.long_exposure / row.equity),
          type: "scatter",
          mode: "lines",
          name: "Long",
          line: { color: palette.teal, width: 1.5 },
        },
        {
          x: dates,
          y: rows.map((row) => -row.short_exposure / row.equity),
          type: "scatter",
          mode: "lines",
          name: "Short",
          line: { color: palette.amber, width: 1.5 },
        },
      ], {
        ...layoutBase,
        yaxis: { ...layoutBase.yaxis, tickformat: ".0%", title: "Share of equity" },
      }, config);
      fillTable(document.getElementById("equity-table-body"), rows, [
        { key: "date", header: true },
        { key: "equity", format: money },
        { key: "benchmark", format: (value) => Number.isFinite(value) ? money(value) : "—" },
      ]);
      fillTable(document.getElementById("drawdown-table-body"), rows, [
        { key: "date", header: true },
        { key: "drawdown", format: percent },
      ]);
      fillTable(document.getElementById("exposure-table-body"), rows, [
        { key: "date", header: true },
        { key: "long_exposure", format: (value, row) => percent(value / row.equity) },
        { key: "short_exposure", format: (value, row) => percent(value / row.equity) },
      ]);
    } catch (error) {
      const status = document.getElementById("page-status");
      if (status) status.textContent = error.message;
    }
  }

  async function renderCompare() {
    const ledger = document.getElementById("compare-ledger");
    if (!ledger || !window.Plotly) return;
    const runIds = JSON.parse(ledger.dataset.runIds || "[]");
    try {
      const series = await Promise.all(
        runIds.map(async (runId) => ({
          runId,
          rows: await getJson(`/api/runs/${encodeURIComponent(runId)}/series?scope=full`),
        })),
      );
      const colors = [palette.teal, palette.amber, palette.ink, palette.red, "#5f6e9c", "#7b6548"];
      const traces = series.map((item, index) => {
        const baseline = item.rows[0]?.equity || 1;
        return {
          x: item.rows.map((row) => row.date),
          y: item.rows.map((row) => (row.equity / baseline) * 100),
          type: "scatter",
          mode: "lines",
          name: item.runId.slice(0, 12),
          line: { color: colors[index % colors.length], width: 2 },
        };
      });
      window.Plotly.newPlot("compare-chart", traces, {
        ...layoutBase,
        yaxis: { ...layoutBase.yaxis, title: "Indexed equity (start = 100)" },
      }, config);

      const monthly = series.map((item) => ({
        runId: item.runId,
        rows: monthEndRows(item.rows),
      }));
      const dates = [...new Set(monthly.flatMap((item) => item.rows.map((row) => row.date)))].sort();
      const lookups = monthly.map((item) => ({
        runId: item.runId,
        values: new Map(item.rows.map((row) => [row.date, row.equity])),
      }));
      const head = document.getElementById("compare-table-head");
      const body = document.getElementById("compare-table-body");
      head.replaceChildren();
      body.replaceChildren();
      const headRow = document.createElement("tr");
      ["Date", ...runIds.map((id) => id.slice(0, 12))].forEach((label) => {
        const th = document.createElement("th");
        th.scope = "col";
        th.textContent = label;
        headRow.append(th);
      });
      head.append(headRow);
      for (const date of dates) {
        const row = document.createElement("tr");
        const dateCell = document.createElement("th");
        dateCell.scope = "row";
        dateCell.textContent = date;
        row.append(dateCell);
        for (const lookup of lookups) {
          const value = lookup.values.get(date);
          const cell = document.createElement("td");
          cell.textContent = value === undefined ? "—" : money(value);
          row.append(cell);
        }
        body.append(row);
      }
    } catch (error) {
      const status = document.getElementById("page-status");
      if (status) status.textContent = error.message;
    }
  }

  window.addEventListener("load", () => {
    renderDetail();
    renderCompare();
  });
})();
