-- Test error handling for invalid SHARED_ONLY regex patterns
-- Should fail with appropriate error messages

DROP TABLE IF EXISTS test_json_shared_only_error;

-- Test invalid regex pattern
CREATE TABLE test_json_shared_only_error (
    id UInt32,
    data JSON(
        user.id UInt32,
        SHARED_ONLY REGEXP '[invalid_regex'
    )
) ENGINE = Memory; -- { serverError CANNOT_COMPILE_REGEXP }

-- Test conflict between typed path and SHARED_ONLY
CREATE TABLE test_json_shared_only_error2 (
    id UInt32,
    data JSON(
        user.id UInt32,
        SHARED_ONLY user.id
    )
) ENGINE = Memory; -- { serverError BAD_ARGUMENTS }
