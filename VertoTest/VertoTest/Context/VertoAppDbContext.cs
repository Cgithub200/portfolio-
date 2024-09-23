using Microsoft.EntityFrameworkCore;
using VertoTest.Models;

namespace VertoTest.Context
{
    public class VertoAppDbContext : DbContext
    {
        public IConfiguration Config { get; set; }
        public VertoAppDbContext (IConfiguration Config)
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

            
            modelBuilder.Entity<StaticContent>().HasKey(sc => sc.ContentId);
            modelBuilder.Entity<StaticContent>().Property(sc => sc.Title).HasMaxLength(200);
           
            modelBuilder.Entity<StaticContent>().HasOne(sc => sc.SliderItem).WithMany(si => si.StaticContents).HasForeignKey(sc => sc.SliderItemID).IsRequired(false).OnDelete(DeleteBehavior.Cascade); 

            modelBuilder.Entity<Image>().HasKey(i => i.ImageId);
            modelBuilder.Entity<Image>().HasOne(i => i.staticContent).WithMany(sc => sc.Images).HasForeignKey(i => i.StaticContentID).IsRequired(false).OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<ContentPanels>().HasKey(si => si.ContentItemID);
   

        }

        public DbSet<StaticContent> StaticContents { get; set; }
        public DbSet<Image> Images { get; set; }
        public DbSet<ContentPanels> ContentPanels { get; set; }

    }
}
