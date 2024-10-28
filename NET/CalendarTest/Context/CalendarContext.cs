using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using VertoTest.Models;

namespace VertoTest.Context
{
    public class CalendarContext : DbContext
    {
        public IConfiguration Config { get; set; }
        public CalendarContext (IConfiguration Config)
        {
            this.Config = Config;
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlServer(this.Config.GetConnectionString("VertoDbConnection"));
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<CalendarEvent>(entity =>
            {
                entity.Property(e => e.LastUpdated).HasDefaultValueSql("GETDATE()");
                entity.HasKey(sc => sc.EventID);
                entity.Property(sc => sc.Title).HasMaxLength(100);
                entity.Property(sc => sc.Description).HasMaxLength(255);
            });
        }

        internal async Task FindAsync(int eventID)
        {
            throw new NotImplementedException();
        }

        public DbSet<CalendarEvent> Events { get; set; }

    }
}
