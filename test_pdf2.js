const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // Expose function to log to Node console
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));

  await page.setContent(`
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    </head>
    <body style="width: 100vw; height: 100vh;">
        <div id="status">Waiting...</div>
        <script>
            setTimeout(async () => {
                const container = document.createElement('div');
                container.innerHTML = "<h1 style='color:red;'>TEST PDF RENDERING</h1><p>hello world</p>";
                container.style.position = 'absolute';
                container.style.left = '-9999px';
                container.style.top = '0';
                container.style.width = '800px';
                document.body.appendChild(container);

                try {
                    const opt = {
                      margin: 0.5,
                      filename: "test.pdf",
                      image: { type: 'jpeg', quality: 0.98 },
                      html2canvas: { scale: 2, useCORS: true, windowWidth: 800 },
                      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
                    };
                    const pdfBase64 = await html2pdf().set(opt).from(container).outputPdf('datauristring');
                    console.log("BASE64 START:", typeof pdfBase64, typeof pdfBase64 === 'string' ? pdfBase64.substring(0, 50) : Object.keys(pdfBase64));
                    // Check if it's blank (very small base64 length might indicate blank)
                    console.log("BASE64 LENGTH:", typeof pdfBase64 === 'string' ? pdfBase64.length : 0);
                    
                    document.getElementById('status').innerText = 'Base64 Length: ' + (typeof pdfBase64 === 'string' ? pdfBase64.length : 0);
                } catch(e) {
                    console.error("ERROR 1:", e.message);
                }
                
                console.log("DONE");
            }, 500);
        </script>
    </body>
    </html>
  `);

  await new Promise(r => setTimeout(r, 3000));
  await browser.close();
})();
