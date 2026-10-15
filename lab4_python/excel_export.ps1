# Opens lab4_excel.xlsx in Microsoft Excel, recalculates it and saves
# PDF pictures of the calculation sheets and of every trendline chart.
# Prints JSON with values computed by Excel itself. Called from lab4.py.
param([Parameter(Mandatory = $true)][string]$Workbook,
      [Parameter(Mandatory = $true)][string]$OutDir)

$ErrorActionPreference = 'Stop'
$xl = New-Object -ComObject Excel.Application
$xl.Visible = $false
$xl.DisplayAlerts = $false
$xl.ScreenUpdating = $true
$result = [ordered]@{ ranges = @(); charts = @(); cells = [ordered]@{} }

function Save-RangePdf($sheet, $address, $file, $landscape) {
    $xl.PrintCommunication = $false
    $setup = $sheet.PageSetup
    $setup.PrintArea = $address
    $setup.Zoom = $false
    $setup.FitToPagesWide = 1
    $setup.FitToPagesTall = 1
    $setup.Orientation = $(if ($landscape) { 2 } else { 1 })
    $setup.PrintGridlines = $true
    $setup.LeftMargin = 10; $setup.RightMargin = 10; $setup.TopMargin = 10; $setup.BottomMargin = 10
    $setup.CenterFooter = ''; $setup.CenterHeader = ''
    $xl.PrintCommunication = $true
    $sheet.ExportAsFixedFormat(0, $file)
}

try {
    $wb = $xl.Workbooks.Open($Workbook)
    $xl.CalculateFull()
    $data = $wb.Worksheets.Item(1)
    $calc = $wb.Worksheets.Item(2)
    $models = $wb.Worksheets.Item(3)
    $charts = $wb.Worksheets.Item(4)

    foreach ($row in 2..32) {
        $name = $calc.Cells.Item($row, 1).Text
        if ($name) { $result.cells["$row " + $name] = $calc.Cells.Item($row, 3).Value2 }
    }
    $plan = @(@($data, 'A1:G31', 'excel_sheet_data.pdf', $false),
              @($calc, 'A1:C32', 'excel_sheet_calc.pdf', $false),
              @($models, 'A1:K10', 'excel_sheet_models.pdf', $true))
    foreach ($item in $plan) {
        $file = Join-Path $OutDir $item[2]
        Save-RangePdf $item[0] $item[1] $file $item[3]
        $result.ranges += $item[2]
    }
    for ($k = 1; $k -le $charts.ChartObjects().Count; $k++) {
        $chart = $charts.ChartObjects($k).Chart
        $label = $chart.SeriesCollection(1).Trendlines(1).DataLabel.Text
        $file = Join-Path $OutDir ("excel_chart_{0}.png" -f $k)
        $null = $chart.Export($file)
        $result.charts += [ordered]@{ file = (Split-Path $file -Leaf); title = $chart.ChartTitle.Text;
                                      label = ($label -replace "`r|`n", ' ; ') }
    }
    $wb.Close($false)
}
finally {
    $xl.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($xl) | Out-Null
}
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$result | ConvertTo-Json -Depth 5 -Compress
