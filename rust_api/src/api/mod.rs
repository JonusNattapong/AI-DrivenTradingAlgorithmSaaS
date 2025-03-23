//! API module for the trading algorithm service

pub mod prediction;

use actix_web::web;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .service(
                web::scope("/v1")
                    .route("/predict", web::post().to(prediction::predict))
                    .route("/health", web::get().to(prediction::health_check))
            )
    );
}