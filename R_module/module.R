# Minimal R equivalents of python_module functionality for tau, gamma estimation

#' Internal: nonlinear equation f(gc, gias, gm_star)
#' @keywords internal
f_gc <- function(gc, gias, gm_star) {
  sqrt(abs(gc * gias / 2)) * tanh(sqrt(abs(2 * gc / gias))) - gm_star
}

#' Internal: derivative df/dgc
#' @keywords internal
df_dgc <- function(gc, gias) {
  0.5 * (
    sqrt(abs(gias / (2 * gc))) * tanh(sqrt(abs(2 * gc / gias))) +
      1 / (gias * cosh(sqrt(abs(2 * gc / gias)))^2)
  )
}

#' Newton solver for cellular conductance gc
#'
#' @param gc_init numeric, initial guess for gc
#' @param gias numeric, intercellular airspace conductance
#' @param gm_star numeric, mesophyll conductance at the compensation point
#' @param step_size numeric, Newton step size multiplier
#' @param max_iterations integer, maximum iterations
#' @param tolerance numeric, convergence tolerance on |f|
#'
#' @return numeric scalar gc estimate
#' @export
newton_solver <- function(gc_init,
                          gias,
                          gm_star,
                          step_size = 0.4,
                          max_iterations = 1000L,
                          tolerance = 1e-6) {
  gc <- gc_init
  for (i in seq_len(max_iterations)) {
    f_val <- f_gc(gc, gias, gm_star)
    if (abs(f_val) < tolerance) {
      return(abs(gc))
    }
    df_val <- df_dgc(gc, gias)
    if (df_val == 0) {
      break
    }
    gc <- gc - step_size * f_val / df_val
  }
  stop("Newton's method did not converge during gc estimation. Consider changing step_size, max_iterations, or initial guess.")
}

#' Estimate intercellular CO2 concentration (Ci) values
#'
#' @param assimilation_rates numeric vector (An)
#' @param stomatal_conductances numeric vector (gs)
#' @param atmospheric_CO2_concentrations numeric vector (Ca)
#'
#' @return numeric vector of Ci estimates (NaN for invalid entries)
#' @export
estimate_Ci_values <- function(assimilation_rates,
                               stomatal_conductances,
                               atmospheric_CO2_concentrations) {
  Ans  <- as.numeric(assimilation_rates)
  gss  <- as.numeric(stomatal_conductances)
  Cas  <- as.numeric(atmospheric_CO2_concentrations)

  if (!(length(Ans) == length(gss) && length(gss) == length(Cas))) {
    stop("All inputs must be one-dimensional vectors of equal length.")
  }

  Cis <- rep(NaN, length(Ans))
  valid <- (Ans >= 0) & (gss > 0) & (Cas > 0) &
    !is.na(Ans) & !is.na(gss) & !is.na(Cas)

  Cis[valid] <- Cas[valid] - (Ans[valid] / gss[valid])
  Cis
}

#' Estimate mesophyll conductance at the CO2 compensation point (gm*)
#'
#' @param assimilation_rates numeric vector (An)
#' @param intercellular_CO2_concentrations numeric vector (Ci)
#' @param compensation_point_CO2_concentrations numeric vector (C_star)
#'
#' @return numeric vector of gm* estimates (NaN for invalid entries)
#' @export
estimate_gm_star_values <- function(assimilation_rates,
                                    intercellular_CO2_concentrations,
                                    compensation_point_CO2_concentrations) {
  Ans     <- as.numeric(assimilation_rates)
  Cis     <- as.numeric(intercellular_CO2_concentrations)
  C_stars <- as.numeric(compensation_point_CO2_concentrations)

  if (!(length(Ans) == length(Cis) && length(Cis) == length(C_stars))) {
    stop("All inputs must be one-dimensional vectors of equal length.")
  }

  gm_stars <- rep(NaN, length(Ans))
  valid <- (Ans >= 0) & (Cis > C_stars) & (C_stars > 0) &
    !is.na(Ans) & !is.na(Cis) & !is.na(C_stars)

  gm_stars[valid] <- Ans[valid] / (Cis[valid] - C_stars[valid])
  gm_stars
}

#' Core numeric entry point: estimate tau and gamma from numeric vectors
#'
#' @param mesophyll_conductances numeric vector gm* (same length as gs, gias)
#' @param stomatal_conductances numeric vector gs
#' @param intercellular_airspace_conductances numeric vector gias
#'
#' @return list with numeric vectors tau and gamma
#' @export
vector_estimate_parameters <- function(mesophyll_conductances,
                                      stomatal_conductances,
                                      intercellular_airspace_conductances) {
  gm_stars <- as.numeric(mesophyll_conductances)
  gss      <- as.numeric(stomatal_conductances)
  giass    <- as.numeric(intercellular_airspace_conductances)

  if (!(length(gm_stars) == length(gss) && length(gss) == length(giass))) {
    stop("All inputs must be one-dimensional vectors of equal length.")
  }

  n <- length(gm_stars)
  gammas <- rep(NaN, n)
  taus   <- rep(NaN, n)

  valid <- (gm_stars > 0) & (gss > 0) & (giass > 0) &
    !is.na(gm_stars) & !is.na(gss) & !is.na(giass)

  gammas[valid] <- 2 * gss[valid] / giass[valid]

  for (i in which(valid)) {
    gm_star <- gm_stars[i]
    gias    <- giass[i]
    gc_init <- gm_star
    gc_est <- tryCatch(
      newton_solver(gc_init, gias, gm_star),
      error = function(e) NaN
    )
    if (is.nan(gc_est)) {
      taus[i] <- NaN
    } else {
      taus[i] <- sqrt(2 * gc_est / gias)
    }
  }

  list(tau = taus, gamma = gammas)
}

#' Helper: require columns in a data.frame
#' @keywords internal
require_columns <- function(required, df) {
  missing <- setdiff(required, colnames(df))
  if (length(missing) > 0L) {
    stop(sprintf("Missing required columns: %s", paste(missing, collapse = ", ")))
  }
}

#' Data frame entry point mirroring pandas_estimate_parameters
#'
#' The data.frame must contain `gs` and `gias`, and either:
#'  - `gm_star`, or
#'  - `An`, `Ci`, `C_star`, or
#'  - `An`, `Ca`, `C_star`.
#'
#' Column names can be remapped via `column_map`, a named list where names are
#' expected names ("gs", "gias", "gm_star", "An", "Ci", "Ca", "C_star")
#' and values are the actual column names in `df`.
#'
#' @param df data.frame with required columns
#' @param column_map named list mapping expected names to column names in df
#'
#' @return data.frame with added numeric columns `tau` and `gamma`
#' @export
pandas_estimate_parameters <- function(df, column_map = NULL) {
  DEFAULT_COL_MAP <- list(
    gs = "gs",
    gias = "gias",
    gm_star = "gm_star",
    An = "An",
    Ci = "Ci",
    Ca = "Ca",
    C_star = "C_star"
  )

  df_ <- as.data.frame(df)

  colmap <- DEFAULT_COL_MAP
  if (!is.null(column_map)) {
    for (nm in names(column_map)) {
      colmap[[nm]] <- column_map[[nm]]
    }
  }

  require_columns(c(colmap$gs, colmap$gias), df_)
  gss   <- as.numeric(df_[[colmap$gs]])
  giass <- as.numeric(df_[[colmap$gias]])

  gm_stars <- rep(NaN, length(gss))
  cols <- colnames(df_)

  if (colmap$gm_star %in% cols) {
    gm_stars <- as.numeric(df_[[colmap$gm_star]])

  } else if (all(c(colmap$An, colmap$Ci, colmap$C_star) %in% cols)) {
    Ans     <- as.numeric(df_[[colmap$An]])
    Cis     <- as.numeric(df_[[colmap$Ci]])
    C_stars <- as.numeric(df_[[colmap$C_star]])
    gm_stars <- estimate_gm_star_values(Ans, Cis, C_stars)

  } else if (all(c(colmap$An, colmap$Ca, colmap$C_star) %in% cols)) {
    Ans     <- as.numeric(df_[[colmap$An]])
    Cas     <- as.numeric(df_[[colmap$Ca]])
    C_stars <- as.numeric(df_[[colmap$C_star]])
    Cis     <- estimate_Ci_values(Ans, gss, Cas)
    gm_stars <- estimate_gm_star_values(Ans, Cis, C_stars)

  } else {
    stop("Data frame must contain either 'gm_star' or 'An', 'Ci', 'C_star' or 'An', 'Ca', 'C_star' columns (after applying column_map, if given).")
  }

  est <- vector_estimate_parameters(gm_stars, gss, giass)
  df_$tau   <- est$tau
  df_$gamma <- est$gamma
  df_
}
