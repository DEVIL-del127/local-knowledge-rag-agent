param([switch]$Apply)
$ErrorActionPreference = 'Stop'

# This maintenance command is intentionally restricted to one known directory.
$taskTarget = 'F:\A_ShiXi\Project\STUDY\agent_state'
$taskResolved = (Resolve-Path -LiteralPath $taskTarget).Path
if ($taskResolved -ne $taskTarget) { throw 'state_path_mismatch' }
$taskSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($taskSid.Value -ne 'S-1-5-21-2024791119-548716333-1742105944-1001') {
    throw 'maintenance_account_mismatch'
}
$taskAllowed = @($taskSid.Value, 'S-1-5-18', 'S-1-5-32-544')
$taskItems = @((Get-Item -LiteralPath $taskResolved)) + @(Get-ChildItem -LiteralPath $taskResolved -Recurse -Force)
foreach ($taskItem in $taskItems) {
    if (($taskItem.FullName -ne $taskResolved) -and -not $taskItem.FullName.StartsWith($taskResolved + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'state_child_path_mismatch'
    }
    if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'state_reparse_point_rejected' }
}

function Test-TaskPrivateAcl($taskPath) {
    $taskAcl = Get-Acl -LiteralPath $taskPath
    $taskOwnerSid = $taskAcl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value
    if ($taskOwnerSid -notin $taskAllowed) { return $false }
    $taskHasUser = $false
    foreach ($taskRule in $taskAcl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])) {
        if ($taskRule.AccessControlType -eq 'Allow' -and [long]$taskRule.FileSystemRights -ne 0) {
            if ($taskRule.IdentityReference.Value -notin $taskAllowed) { return $false }
            if ($taskRule.IdentityReference.Value -eq $taskSid.Value) { $taskHasUser = $true }
        }
    }
    return $taskHasUser
}

if ($Apply) {
    # SDDL contains permissions only, not stored application data or key bytes.
    $taskBackup = @($taskItems | ForEach-Object {
        [pscustomobject]@{Path=$_.FullName; Sddl=(Get-Acl -LiteralPath $_.FullName).Sddl}
    })
    $taskBackupPath = Join-Path $taskResolved ('acl-before-' + [guid]::NewGuid().ToString('N') + '.json')
    # Save the recovery inventory before the first permission mutation.
    $taskBytes = [Text.Encoding]::UTF8.GetBytes(($taskBackup | ConvertTo-Json -Depth 4))
    $taskStream = [IO.File]::Open($taskBackupPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $taskStream.Write($taskBytes, 0, $taskBytes.Length); $taskStream.Flush($true) } finally { $taskStream.Dispose() }
    $taskItems += Get-Item -LiteralPath $taskBackupPath
    foreach ($taskItem in $taskItems) {
        if ($taskItem.PSIsContainer) {
            $taskAcl = New-Object System.Security.AccessControl.DirectorySecurity
            $taskInheritance = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
        } else {
            $taskAcl = New-Object System.Security.AccessControl.FileSecurity
            $taskInheritance = [System.Security.AccessControl.InheritanceFlags]::None
        }
        $taskAcl.SetOwner($taskSid)
        $taskAcl.SetAccessRuleProtection($true, $false)
        foreach ($taskPrincipal in $taskAllowed) {
            $taskRule = [System.Security.AccessControl.FileSystemAccessRule]::new(
                [System.Security.Principal.SecurityIdentifier]::new($taskPrincipal),
                [System.Security.AccessControl.FileSystemRights]::FullControl, $taskInheritance,
                [System.Security.AccessControl.PropagationFlags]::None,
                [System.Security.AccessControl.AccessControlType]::Allow)
            $taskAcl.AddAccessRule($taskRule)
        }
        Set-Acl -LiteralPath $taskItem.FullName -AclObject $taskAcl
    }
    # A partial failure is never a success; the saved inventory supports manual recovery.
    Write-Output ('acl_backup=' + $taskBackupPath)
}
$taskFailed = @($taskItems | Where-Object { -not (Test-TaskPrivateAcl $_.FullName) })
[pscustomobject]@{Checked=$taskItems.Count; Failed=$taskFailed.Count; Private=($taskFailed.Count -eq 0)} | ConvertTo-Json -Compress
if ($taskFailed.Count) { throw 'state_acl_verification_failed' }
