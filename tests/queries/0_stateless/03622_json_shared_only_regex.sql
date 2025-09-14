-- Test SHARED_ONLY with regex patterns
-- SHARED_ONLY REGEXP should match paths using regular expressions

DROP TABLE IF EXISTS test_json_shared_only_regex;

CREATE TABLE test_json_shared_only_regex (
    id UInt32,
    data JSON(
        max_dynamic_paths=3,
        user.id UInt32,
        SHARED_ONLY REGEXP '^meta\\..*',
        SHARED_ONLY REGEXP '^debug\\..*',
        SHARED_ONLY REGEXP '^audit\\..*'
    )
) ENGINE = Memory;

-- Insert data with regex-matched paths
INSERT INTO test_json_shared_only_regex VALUES 
(1, '{"user": {"id": 1}, "meta": {"timestamp": "2023-01-01"}, "debug": {"level": "info"}, "audit": {"action": "login"}, "regular1": "value1"}'),
(2, '{"user": {"id": 2}, "meta": {"version": "1.0"}, "debug": {"trace": "abc123"}, "audit": {"ip": "192.168.1.1"}, "regular1": "value2"}');

-- Verify shared and dynamic paths
SELECT 
    length(JSONDynamicPaths(data)) AS dynamic_path_count,
    length(JSONSharedDataPaths(data)) AS shared_path_count,
    JSONDynamicPaths(data) AS dynamic_paths,
    JSONSharedDataPaths(data) AS shared_paths
FROM test_json_shared_only_regex
ORDER BY id;

-- Verify that regex-matched paths are accessible
SELECT 
    data.user.id as user_id,
    data.meta.timestamp as meta_timestamp,
    data.debug.level as debug_level,
    data.audit.action as audit_action,
    data.regular1 as regular1
FROM test_json_shared_only_regex
ORDER BY id LIMIT 1;


DROP TABLE test_json_shared_only_regex;
