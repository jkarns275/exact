<?php

### Aaron did the following temporarily:
### I commented out lines 51-56, & 109
### on June 25 at 2:29pm


$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

$databases = array(
    "exact_batchnorm" => array(
        "file" => $cwd[__FILE__] . '/../exact_batchnorm_db_info',
        "database" => NULL
    ),
    "exact_bn_pool" => array(
        "file" => $cwd[__FILE__] . '/../exact_bn_pool_db_info',
        "database" => NULL
    ),
    "exact_bn_sfmp" => array(
        "file" => $cwd[__FILE__] . '/../exact_bn_sfmp_db_info',
        "database" => NULL
    )
);

function multi_db_connect($db_name) {
    global $databases;

    $file = file_get_contents($databases[$db_name]["file"]);
    $lines = explode("\n", $file);
    $server = $lines[0];
    $db_name = $lines[1];
    $user = $lines[2];
    $passwd = $lines[3];

    $dbcnx = new mysqli($server, $user, $passwd, $db_name);

    if ($dbcnx->connect_errno) {
        //echo "Failed to connect to MySQL: (" . $dbcnx->connect_errno . ") " . $dbcnx->connect_error;
        error_log("Failed to connect to MySQL: (" . $dbcnx->connect_errno . ") " . $dbcnx->connect_error);
    }

    return $dbcnx;
}

function query_multi_db($db_name, $query) {
    global $databases;

    if ($databases[$db_name]["database"] == NULL || !$databases[$db_name]["database"]->ping()) $databases[$db_name]["database"] = multi_db_connect($db_name);

    $result = $databases[$db_name]["database"]->query($query);

    if (!$result) mysqli_error_msg($databases[$db_name]["database"], $query);

    return $result;
}

?>
