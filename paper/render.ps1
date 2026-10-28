# Convert a .docx to PDF with Microsoft Word (for checking the layout).
# Usage: powershell -ExecutionPolicy Bypass -File paper\render.ps1 paper\ICALT2027_draft_guide.docx
param([string]$Docx)
$full = (Resolve-Path $Docx).Path
$pdf = [System.IO.Path]::ChangeExtension($full, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($full, $false, $true)
    $doc.SaveAs([ref]$pdf, [ref]17)   # 17 = wdFormatPDF
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close([ref]$false)
} finally {
    $word.Quit()
}
"pdf: $pdf"
