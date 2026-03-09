const puppeteer = require('puppeteer');

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
    <body>
        <div id="content">Hello World</div>
        <script>
            setTimeout(async () => {
                try {
                    const obj = await html2pdf().from(document.getElementById('content')).output('datauristring');
                    console.log("BASE64 START:", typeof obj, typeof obj === 'string' ? obj.substring(0, 50) : Object.keys(obj));
                } catch(e) {
                    console.error("ERROR 1:", e.message);
                }

                try {
                    const obj2 = await html2pdf().from(document.getElementById('content')).outputPdf('datauristring');
                    console.log("PDF2:", typeof obj2, typeof obj2 === 'string' ? obj2.substring(0, 50) : Object.keys(obj2));
                } catch(e) {
                    console.error("ERROR 2:", e.message);
                }
                
                console.log("DONE");
            }, 1000);
        </script>
    </body>
    </html>
  `);

  await new Promise(r => setTimeout(r, 4000));
  await browser.close();
})();
