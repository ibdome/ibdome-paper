#' @export
db_connect <- function(sqlite_path) {
    DBI::dbConnect(
        RSQLite::SQLite(),
        sqlite_path,
        flags = RSQLite::SQLITE_RO
    )
}
