#!/usr/bin/env Rscript
# Export aggregate KM estimates without participant identifiers.
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) stop("Missing ", flag)
  args[i + 1]
}
input <- get_arg("--input")
out <- get_arg("--out")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(input)
stopifnot(all(c("BST2", "time", "event") %in% names(d)))
d$group <- factor(ifelse(d$BST2 > median(d$BST2), "BST2 high", "BST2 low"),
                  levels = c("BST2 low", "BST2 high"))
fit <- survival::survfit(survival::Surv(time, event) ~ group, data = d)
sf <- summary(fit)
z <- data.frame(time = sf$time, surv = sf$surv, lower = sf$lower,
                upper = sf$upper, group = as.character(sf$strata))
z$group <- sub("^group=", "", z$group)
z <- do.call(rbind, lapply(c("BST2 low", "BST2 high"), function(g) {
  rbind(data.frame(time = 0, surv = 1, lower = 1, upper = 1, group = g),
        z[z$group == g, ])
}))
write.csv(z, file.path(out, "bst2_km_curve.csv"), row.names = FALSE)
lr <- survival::survdiff(survival::Surv(time, event) ~ group, data = d)
p_logrank <- pchisq(lr$chisq, df = 1, lower.tail = FALSE)
jsonlite::write_json(list(p_logrank = p_logrank, n = nrow(d), events = sum(d$event)),
                     file.path(out, "bst2_km_logrank.json"), auto_unbox = TRUE, digits = 16)
cat("KM aggregate:", nrow(d), "participants,", sum(d$event), "events; log-rank p =", p_logrank, "\n")
