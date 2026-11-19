(() => {
  const status = document.getElementById("page-status");

  function announce(message) {
    if (status) status.textContent = message;
  }

  const compareBoxes = [...document.querySelectorAll(".compare-checkbox")];
  const compareForm = document.getElementById("compare-form");
  if (compareForm) {
    compareForm.addEventListener("submit", (event) => {
      const checked = compareBoxes.filter((box) => box.checked);
      if (checked.length < 2 || checked.length > 6) {
        event.preventDefault();
        announce("Select between 2 and 6 completed runs before comparing.");
        const hint = document.getElementById("compare-hint");
        if (hint) {
          hint.textContent = `Select 2 to 6 runs. You have selected ${checked.length}.`;
          hint.classList.add("field-error");
        }
        const first = checked[0] || compareBoxes[0];
        if (first) first.focus();
      }
    });
  }

  document.querySelectorAll(".archive-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const runId = button.dataset.runId;
      const archived = button.dataset.archived === "true";
      button.disabled = true;
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/archive`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ archived: !archived }),
        });
        if (!response.ok) throw new Error("Archive update failed.");
        button.dataset.archived = String(!archived);
        button.textContent = archived ? "Archive" : "Unarchive";
        if (location.pathname.startsWith(`/runs/${runId}`)) {
          button.textContent = archived ? "Archive run" : "Unarchive run";
          const state = document.getElementById("archive-state");
          if (state) state.textContent = archived ? "Active" : "Archived";
        }
        announce(archived ? "Run unarchived." : "Run archived.");
      } catch (error) {
        announce(error.message);
      } finally {
        button.disabled = false;
      }
    });
  });

  const dialog = document.getElementById("delete-dialog");
  const confirmDelete = document.getElementById("delete-confirm");
  let pendingRunId = null;
  document.querySelectorAll(".delete-trigger").forEach((button) => {
    button.addEventListener("click", () => {
      pendingRunId = button.dataset.runId;
      const runName = document.getElementById("delete-run-name");
      if (runName) runName.textContent = pendingRunId;
      dialog?.showModal();
    });
  });

  if (confirmDelete) {
    confirmDelete.addEventListener("click", async () => {
      if (!pendingRunId) return;
      confirmDelete.disabled = true;
      try {
        const response = await fetch(
          `/api/runs/${encodeURIComponent(pendingRunId)}?confirm=true`,
          { method: "DELETE" },
        );
        if (!response.ok) throw new Error("Run deletion failed.");
        announce("Run deleted.");
        dialog?.close();
        window.location.href = "/";
      } catch (error) {
        announce(error.message);
        confirmDelete.disabled = false;
      }
    });
  }
})();
