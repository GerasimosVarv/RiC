$ErrorActionPreference = "Stop"

$Files = @(
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\DE_CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\EUA_CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\FR_CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\GR_CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\HU_CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\ICE_PSV_CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\ICE_TTF__CURVE.xlsx",
    "C:\Users\g.varvounis\OneDrive - HELLENiQ ENERGY\Risk Section Activity\RiC Monitoring\_data\BBG\ICE_THE_CURVE.xlsx"
)

$LogFile = "C:\Scripts\BloombergRefresh.log"
$Excel = $null

try {
    Add-Content $LogFile "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Refresh started"

    $Excel = New-Object -ComObject Excel.Application
    $Excel.Visible = $false
    $Excel.DisplayAlerts = $false
    $Excel.AskToUpdateLinks = $false
    $Excel.EnableEvents = $true
    $Excel.Calculation = -4105  # xlCalculationAutomatic

    foreach ($File in $Files) {
        Add-Content $LogFile "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Opening $File"

        $Workbook = $Excel.Workbooks.Open($File)

        # Recalculate TODAY()-1 and all Bloomberg BDH formulas
        $Excel.CalculateFullRebuild()

        # Give Bloomberg time to return the data
        $TimeoutSeconds = 180
        $Elapsed = 0

        do {
            Start-Sleep -Seconds 5
            $Elapsed += 5
            $Excel.Calculate()

            $BloombergPending = $false

            foreach ($Worksheet in $Workbook.Worksheets) {
                $Values = $Worksheet.UsedRange.Value2

                if ($Values -is [Array]) {
                    foreach ($Value in $Values) {
                        if (
                            $Value -like "*Requesting Data*" -or
                            $Value -like "*#N/A Requesting Data*" -or
                            $Value -like "*#N/A Connection*"
                        ) {
                            $BloombergPending = $true
                            break
                        }
                    }
                }

                if ($BloombergPending) {
                    break
                }
            }
        }
        while ($BloombergPending -and $Elapsed -lt $TimeoutSeconds)

        if ($BloombergPending) {
            Add-Content $LogFile "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - WARNING: Bloomberg timeout for $File"
        }
        else {
            $Workbook.Save()
            Add-Content $LogFile "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Successfully refreshed $File"
        }

        $Workbook.Close($false)
    }

    Add-Content $LogFile "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Refresh completed"
}
catch {
    Add-Content $LogFile "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - ERROR: $($_.Exception.Message)"
    throw
}
finally {
    if ($Excel -ne $null) {
        $Excel.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($Excel) | Out-Null
    }

    :Collect()
    :WaitForPendingFinalizers()
}