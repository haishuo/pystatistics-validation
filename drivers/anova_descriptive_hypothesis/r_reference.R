#!/usr/bin/env Rscript
# R reference dispatcher for anova / descriptive / hypothesis validation.
#
# One job: read a JSON job file {"func": ..., <args>}, compute the canonical R
# reference with the MATCHING convention, and emit a flat JSON dict of the
# comparable quantities on stdout. The driver feeds the identical fp64 numbers
# pystatistics analyses (shared-input discipline), so agreement is fp64
# round-off of the same inputs.
#
# References per function:
#   var/cov/cor/quantile  -> base stats (quantile type matched explicitly)
#   t/chisq/fisher/wilcox/ks/prop/var.test/p.adjust -> base stats
#   anova_oneway          -> anova(lm(...)) sequential Type I
#   anova (factorial)     -> car::Anova(type="II"/"III") ; Type I via anova(lm)
#   levene                -> car::leveneTest (center=median default)
#   posthoc tukey         -> TukeyHSD(aov(...))
#   anova_rm              -> afex::aov_ez (Mauchly + GG/HF), cross-checked base

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)
job <- fromJSON(args[[1]], simplifyVector = TRUE)
f <- job$func

# pystatistics spells it "two-sided"; base R wants "two.sided".
if (!is.null(job$alternative)) job$alternative <- gsub("-", ".", job$alternative)

# jsonlite decodes a JSON array of numbers to a numeric vector; a scalar stays
# length-1. Factor/label columns arrive as character vectors.
num <- function(x) as.numeric(x)

# jsonlite reads a JSON array-of-arrays as a row-oriented matrix already, so a
# 2-D table arrives correctly shaped; only a flat vector (GoF) needs reshaping.
as_table <- function(job) {
  x <- job$table
  if (is.matrix(x)) return(x)
  matrix(num(x), nrow = job$nrow, byrow = TRUE)
}

# jsonlite cannot emit Inf/-Inf/NaN as JSON numbers; encode them as tagged
# strings so the Python bridge can restore the float (conf-int / odds-ratio
# bounds legitimately go to +/-Inf).
fix_inf <- function(x) {
  if (is.list(x)) return(lapply(x, fix_inf))
  if (is.numeric(x) && any(!is.finite(x))) {
    return(lapply(as.list(x), function(v)
      if (is.finite(v)) v
      else if (is.nan(v)) "__NaN__"
      else if (v > 0) "__Inf__" else "__-Inf__"))
  }
  x
}

emit <- function(lst) {
  cat(toJSON(fix_inf(lst), digits = NA, auto_unbox = TRUE, na = "null"))
}

# Capture warnings so the driver can match R's warning behaviour (R10/G2).
warns <- character(0)
wh <- function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }

res <- withCallingHandlers({

if (f == "var") {
  list(value = var(num(job$x)))

} else if (f == "sd") {
  list(value = sd(num(job$x)))

} else if (f == "cov_pair") {
  list(value = cov(num(job$x), num(job$y)))

} else if (f == "cor_pair") {
  list(value = cor(num(job$x), num(job$y), method = job$method))

} else if (f == "cov_matrix") {
  X <- matrix(num(job$x), nrow = job$nrow, byrow = TRUE)
  list(value = as.vector(cov(X)))

} else if (f == "cor_matrix") {
  X <- matrix(num(job$x), nrow = job$nrow, byrow = TRUE)
  list(value = as.vector(cor(X, method = job$method)))

} else if (f == "quantile") {
  q <- quantile(num(job$x), probs = num(job$probs), type = job$type, names = FALSE)
  list(value = as.vector(q))

} else if (f == "moments") {
  suppressMessages(library(e1071))
  x <- num(job$x)
  list(mean = mean(x), var = var(x), sd = sd(x),
       min = min(x), max = max(x), median = median(x),
       skewness = e1071::skewness(x, type = 2),
       kurtosis = e1071::kurtosis(x, type = 2),
       n = length(x))

} else if (f == "t_test") {
  x <- num(job$x)
  a <- list(x = x, alternative = job$alternative, mu = job$mu,
            conf.level = job$conf_level)
  if (!is.null(job$y)) {
    a$y <- num(job$y); a$paired <- isTRUE(job$paired)
    a$var.equal <- isTRUE(job$var_equal)
  }
  r <- do.call(t.test, a)
  list(statistic = unname(r$statistic), df = unname(r$parameter),
       p_value = r$p.value, conf_int = as.vector(r$conf.int),
       estimate = as.vector(r$estimate))

} else if (f == "var_test") {
  r <- var.test(num(job$x), num(job$y), ratio = job$ratio,
                alternative = job$alternative, conf.level = job$conf_level)
  list(statistic = unname(r$statistic), df1 = unname(r$parameter[1]),
       df2 = unname(r$parameter[2]), p_value = r$p.value,
       conf_int = as.vector(r$conf.int), estimate = unname(r$estimate))

} else if (f == "prop_test") {
  r <- prop.test(x = num(job$x), n = num(job$n),
                 p = if (is.null(job$p)) NULL else num(job$p),
                 alternative = job$alternative, conf.level = job$conf_level,
                 correct = isTRUE(job$correct))
  list(statistic = unname(r$statistic), df = unname(r$parameter),
       p_value = r$p.value, conf_int = as.vector(r$conf.int),
       estimate = as.vector(r$estimate))

} else if (f == "chisq_test") {
  tab <- as_table(job)
  if (!is.null(job$p)) {
    r <- chisq.test(num(job$table), p = num(job$p))   # goodness-of-fit
  } else {
    r <- chisq.test(tab, correct = isTRUE(job$correct))
  }
  list(statistic = unname(r$statistic), df = unname(r$parameter),
       p_value = r$p.value, expected = as.vector(t(r$expected)))

} else if (f == "fisher_test") {
  tab <- as_table(job)
  a <- list(x = tab, alternative = job$alternative, workspace = 2e8)
  if (!is.null(job$conf_level)) a$conf.level <- job$conf_level
  r <- do.call(fisher.test, a)
  out <- list(p_value = r$p.value)
  if (!is.null(r$estimate)) out$estimate <- unname(r$estimate)
  if (!is.null(r$conf.int)) out$conf_int <- as.vector(r$conf.int)
  out

} else if (f == "wilcox_test") {
  x <- num(job$x)
  a <- list(x = x, alternative = job$alternative, mu = job$mu,
            correct = isTRUE(job$correct), conf.int = isTRUE(job$conf_int),
            conf.level = job$conf_level)
  if (!is.null(job$exact)) a$exact <- isTRUE(job$exact)
  if (!is.null(job$y)) { a$y <- num(job$y); a$paired <- isTRUE(job$paired) }
  r <- do.call(wilcox.test, a)
  out <- list(statistic = unname(r$statistic), p_value = r$p.value)
  if (!is.null(r$conf.int)) out$conf_int <- as.vector(r$conf.int)
  if (!is.null(r$estimate)) out$estimate <- unname(r$estimate)
  out

} else if (f == "ks_test") {
  x <- num(job$x)
  if (!is.null(job$y)) {
    r <- ks.test(x, num(job$y), alternative = job$alternative)
  } else {
    r <- ks.test(x, job$dist, job$arg1, job$arg2, alternative = job$alternative)
  }
  list(statistic = unname(r$statistic), p_value = r$p.value)

} else if (f == "p_adjust") {
  list(value = as.vector(p.adjust(num(job$p), method = job$method)))

} else if (f == "anova_oneway") {
  y <- num(job$y); g <- factor(job$group)
  # Type I sequential (single factor: I=II=III coincide)
  m <- lm(y ~ g)
  at <- anova(m)
  list(df = at$Df[1], ss = at$`Sum Sq`[1], ms = at$`Mean Sq`[1],
       F = at$`F value`[1], p_value = at$`Pr(>F)`[1],
       df_resid = at$Df[2], ss_resid = at$`Sum Sq`[2])

} else if (f == "anova_factorial") {
  y <- num(job$y)
  fac <- job$factors            # named list of character vectors
  df <- data.frame(y = y)
  for (nm in names(fac)) df[[nm]] <- factor(fac[[nm]])
  form <- as.formula(paste("y ~", job$formula_rhs))
  ss_type <- job$ss_type
  if (ss_type == 1) {
    m <- lm(form, data = df); at <- anova(m)
    terms <- rownames(at)
    list(terms = terms[-length(terms)],
         df = at$Df[-length(terms)], ss = at$`Sum Sq`[-length(terms)],
         F = at$`F value`[-length(terms)], p_value = at$`Pr(>F)`[-length(terms)],
         df_resid = at$Df[length(terms)], ss_resid = at$`Sum Sq`[length(terms)])
  } else {
    suppressMessages(library(car))
    contr <- if (ss_type == 3) list(unordered = "contr.sum", ordered = "contr.poly") else NULL
    m <- lm(form, data = df,
            contrasts = if (ss_type == 3) {
              cl <- list(); for (nm in names(fac)) cl[[nm]] <- "contr.sum"; cl
            } else NULL)
    at <- car::Anova(m, type = ss_type)
    # car::Anova rows: terms then Residuals; type III also has (Intercept)
    rn <- rownames(at)
    ridx <- which(rn == "Residuals")
    keep <- setdiff(seq_along(rn), ridx)
    if (ss_type == 3) keep <- keep[rn[keep] != "(Intercept)"]
    list(terms = rn[keep], df = at$Df[keep], ss = at$`Sum Sq`[keep],
         F = at$`F value`[keep], p_value = at$`Pr(>F)`[keep],
         df_resid = at$Df[ridx], ss_resid = at$`Sum Sq`[ridx])
  }

} else if (f == "levene") {
  suppressMessages(library(car))
  y <- num(job$y); g <- factor(job$group)
  r <- car::leveneTest(y, g, center = job$center)
  list(df1 = r$Df[1], df2 = r$Df[2], F = r$`F value`[1], p_value = r$`Pr(>F)`[1])

} else if (f == "tukey") {
  y <- num(job$y); g <- factor(job$group)
  m <- aov(y ~ g)
  r <- TukeyHSD(m, conf.level = job$conf_level)$g
  list(comparisons = rownames(r), diff = r[, "diff"], lwr = r[, "lwr"],
       upr = r[, "upr"], p_adj = r[, "p adj"])

} else if (f == "anova_rm") {
  suppressMessages(library(afex))
  # jsonlite already delivers the n x k matrix row-oriented; do NOT reshape.
  Y <- if (is.matrix(job$Y)) job$Y else matrix(num(job$Y), nrow = job$n, byrow = TRUE)
  n <- job$n; k <- job$k
  long <- data.frame(
    id = factor(rep(seq_len(n), times = k)),
    cond = factor(rep(seq_len(k), each = n)),
    y = as.vector(Y)                                     # column-major: cond-major
  )
  a <- suppressWarnings(afex::aov_ez("id", "y", long, within = "cond",
                                     anova_table = list(correction = "none")))
  # within-subject F (uncorrected), plus Mauchly + GG/HF epsilons
  gg <- summary(a)$pval.adjustments
  tab <- a$anova_table
  list(df1 = tab$`num Df`[1], df2 = tab$`den Df`[1], F = tab$F[1],
       p_value = tab$`Pr(>F)`[1],
       gg_eps = gg[1, "GG eps"], hf_eps = gg[1, "HF eps"],
       p_gg = gg[1, "Pr(>F[GG])"], p_hf = gg[1, "Pr(>F[HF])"],
       mauchly_W = suppressWarnings(summary(a)$sphericity.tests[1, "Test statistic"]),
       mauchly_p = suppressWarnings(summary(a)$sphericity.tests[1, "p-value"]))

} else {
  stop(paste("unknown func:", f))
}

}, warning = wh)

res$r_warnings <- if (length(warns)) warns else NULL
emit(res)
