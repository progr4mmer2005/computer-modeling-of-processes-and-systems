# Fills the table of contents (and page-number fields) of a .docx report with Microsoft Word and saves it.
param([Parameter(Mandatory = $true)][string]$Docx)
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $null = $doc.Fields.Update()
    $doc.Save()
    "toc: " + $doc.TablesOfContents.Count + ", pages: " + $doc.ComputeStatistics(2)
    $doc.Close($false)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
