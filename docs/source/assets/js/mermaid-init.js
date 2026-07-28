document.addEventListener("DOMContentLoaded", function () {
  // Convert <pre class="mermaid"><code>...</code></pre> to <div class="mermaid">...</div>
  var pres = document.querySelectorAll("pre.mermaid");
  for (var i = 0; i < pres.length; i++) {
    var pre = pres[i];
    var code = pre.querySelector("code");
    var text = code ? code.textContent : pre.textContent;
    var div = document.createElement("div");
    div.className = "mermaid";
    div.textContent = text;
    pre.parentNode.replaceChild(div, pre);
  }
  // Initialize and render
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      securityLevel: "loose"
    });
    try {
      mermaid.run({ querySelector: ".mermaid" });
    } catch (e) {
      console.error("Mermaid render error:", e);
    }
  } else {
    console.error("mermaid is not defined");
  }
});
